#include <iostream>
#include <chrono>
#include <thread>
#include <unordered_set>
#include "../../include/utils/DistributedManager.h"

DistributedManager::DistributedManager(MetadataStore* metadata_store, PrefetcherBackend** prefetcher_backends) {
    this->metadata_store = metadata_store;
    pf_backends = prefetcher_backends;
    int initialized; // If multiple jobs run in parallel or MPI was already initialized by e.g. Horovod, initialize only once
    MPI_Initialized(&initialized);
    int provided;
    if (!initialized) {
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        has_initialized_mpi = true;
    } else {
        MPI_Query_thread(&provided);
    }
    if (provided < MPI_THREAD_MULTIPLE) {
        throw std::runtime_error("Implementation initialized without MPI_THREAD_MULTIPLE");
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &JOB_COMM);
    MPI_Comm_size(JOB_COMM, &n);
    MPI_Comm_rank(JOB_COMM, &node_id);
    stop_flag = false;
}

void DistributedManager::set_backend(StorageBackend* backend) {
    storage_backend = backend;
}

int DistributedManager::get_no_nodes() const {
    return n;
}

int DistributedManager::get_node_id() const {
    return node_id;
}

void DistributedManager::serve() {
  constexpr int num_comms = 16;
  MPI_Request reqs[num_comms*2];
  MPI_Status statuses[num_comms*2];
  int indices[num_comms*2];
  int recv_bufs[num_comms*2];  // Requested file ID and tag.
  for (int i = 0; i < num_comms*2; ++i) {
    reqs[i] = MPI_REQUEST_NULL;
  }
  // Post initial receives.
  for (int i = 0; i < num_comms; ++i) {
    MPI_Irecv(&recv_bufs[i*2], 2, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG,
              JOB_COMM, &reqs[i]);
  }
  while (true) {
    int completed_reqs = 0;
    MPI_Waitsome(num_comms*2, reqs, &completed_reqs, indices, statuses);
    // If asked to stop, just bail. (TODO Fix.)
    if (stop_flag.load()) {
      break;
    }
    for (int i = 0; i < completed_reqs; ++i) {
      int op_idx = indices[i];
      if (op_idx < num_comms) {
        // Received a request for a file_id. Send it.
        int send_idx = op_idx + num_comms;
        int file_id = recv_bufs[op_idx*2];
        int tag = recv_bufs[op_idx*2+1];
        int source = statuses[i].MPI_SOURCE;
        int storage_level = metadata_store->get_storage_level(file_id);
        if (storage_level > 0) {
          unsigned long len;
          char* buf = pf_backends[storage_level - 1]->get_location(file_id, &len);
          MPI_Isend(buf, (int) len, MPI_CHAR, source, tag,
                    JOB_COMM, &reqs[send_idx]);
        } else {
          // Sample not available. Send zero bytes.
          MPI_Isend(nullptr, 0, MPI_CHAR, source, tag,
                    JOB_COMM, &reqs[send_idx]);
        }
      } else {
        // Completed a send. Post another receive.
        int recv_idx = op_idx - num_comms;
        MPI_Irecv(&recv_bufs[recv_idx*2], 2, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG,
                  JOB_COMM, &reqs[recv_idx]);
      }
    }
  }
}

/**
 * @return true if fetching was succesful, false otherwise
 */
bool DistributedManager::fetch(int file_id, char* dst, int thread_id) {
    unsigned long file_size = storage_backend->get_file_size(file_id);
    int from_node;
    if (file_availability.count(file_id) != 0) {
        from_node = file_availability[file_id].node_id;
    } else {
        return false;
    }
    // Send requested file id and answer tag
    int tag = thread_id + 1;
    int req[2] = {file_id, tag};
    MPI_Send(&req, 2, MPI_INT, from_node, REQUEST_TAG, JOB_COMM);
    MPI_Status response_status;
    MPI_Recv(dst, (int) file_size, MPI_CHAR, from_node, tag, JOB_COMM, &response_status);
    int response_length;
    MPI_Get_count(&response_status, MPI_CHAR, &response_length);
    return response_length > 0;
}

void DistributedManager::distribute_prefetch_strings(std::vector<int>* local_prefetch_string,
                                                     std::vector<std::vector<int>::iterator>* storage_class_ends,
                                                     int num_storage_classes) {
    int local_size = local_prefetch_string->size();
    int global_max_size;
    MPI_Allreduce(&local_size, &global_max_size, 1, MPI_INT, MPI_MAX, JOB_COMM);
    // We send at most (num_storage_classes - 1) length values, total size (including used number of storage classes) therefore
    int arr_size = global_max_size + num_storage_classes;
    int send_data[arr_size];
    int rcv_data[n * arr_size];
    for (int i = 0; i < local_size; i++) {
        send_data[i] = (*local_prefetch_string)[i];
    }
    auto prev_end = local_prefetch_string->begin();
    // Store number of elements per storage class in send_data
    send_data[global_max_size] = storage_class_ends->size();
    for (unsigned long i = 0; i < storage_class_ends->size(); i++) {
        send_data[global_max_size + i + 1] = std::distance(prev_end, (*storage_class_ends)[i]);
        prev_end = (*storage_class_ends)[i];
    }
    MPI_Datatype arr_type;
    MPI_Type_contiguous(arr_size, MPI_INT, &arr_type);
    MPI_Type_commit(&arr_type);
    MPI_Allgather(&send_data, 1, arr_type, rcv_data, 1, arr_type, JOB_COMM);
    MPI_Type_free(&arr_type);
    parse_received_prefetch_data(rcv_data, arr_size, global_max_size);
}

void DistributedManager::parse_received_prefetch_data(int* rcv_data, int arr_size, int global_max_size) {
    for (int i = 0; i < n; i++) {
        if (i != node_id) {
            int offset = i * arr_size;
            int used_storage_classes = rcv_data[offset + global_max_size];
            std::vector<int> elems_per_storage_class;
            for (int j = offset + global_max_size + 1; j < offset + global_max_size + 1 + used_storage_classes; j++) {
                elems_per_storage_class.push_back(rcv_data[j]);
            }
            for (unsigned long j = 0; j < elems_per_storage_class.size(); j++) {
                int storage_class_elems = elems_per_storage_class[j];
                for (int k = offset; k < offset + storage_class_elems; k++) {
                    int file_id = rcv_data[k];
                    struct FileAvailability file_avail{};
                    file_avail.node_id = i;
                    file_avail.offset = k - offset;
                    file_avail.storage_class = j + 1;
                    if (file_availability.count(file_id) > 0) {
                        if (file_avail.storage_class < file_availability[file_id].storage_class ||
                        (file_avail.storage_class == file_availability[file_id].storage_class && file_avail.offset < file_availability[file_id].offset)) {
                            file_availability[file_id] = file_avail;
                        }
                    } else {
                        file_availability[file_id] = file_avail;
                    }
                }
                offset += storage_class_elems;
            }
        }
    }
}

int DistributedManager::generate_and_broadcast_seed() {
    int seed;
    if (node_id == 0) {
        seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    MPI_Bcast(&seed, 1, MPI_INT, 0, JOB_COMM);
    return seed;
}

/**
 * Sets storage class to the lowest available remote storage class, 0 if none available
 */
int DistributedManager::get_remote_storage_class(int file_id) {
    if (file_availability.count(file_id) == 0) {
        return 0;
    }
    FileAvailability fa = file_availability[file_id];
    int remote_offset = fa.offset;
    int remote_storage_class = fa.storage_class;
    /**
     * We use the following heuristic to decide that a file should be available at a remote node (note that wrong decisions
     * don't lead to an error, as the fetching logic will detect these cases, but it will hurt performance):
     * Case 1: Storage class only exists on remote node which means that it contains only a few entries, we therefore assume the file is available.
     * Case 2: We're ahead by at least REMOTE_PREFETCH_OFFSET_DIFF in our local prefetch string and can therefore assume that the other node is at least at remote_offset
     * Case 3: We're done prefetching and can therefore assume the other node is as well.
     */
    if (pf_backends[remote_storage_class - 1] == nullptr ||
        pf_backends[remote_storage_class - 1]->get_prefetch_offset() > remote_offset + REMOTE_PREFETCH_OFFSET_DIFF ||
        pf_backends[remote_storage_class - 1]->is_done()) {
        return remote_storage_class;
    }
    return 0;
}

void DistributedManager::stop_all_threads(int num_threads) {
    // TODO Does not work with multiple threads.
    stop_flag.store(true);
    for (int i = 0; i < num_threads; i++) {
        MPI_Send(nullptr, 0, MPI_INT, node_id, REQUEST_TAG, JOB_COMM);
    }
    MPI_Comm_free(&JOB_COMM);
}
