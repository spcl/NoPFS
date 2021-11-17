#include <iostream>
#include <thread>
#include <cstring>
#include "../../include/prefetcher/StagingBufferPrefetcher.h"
#include "../../include/utils/Metrics.h"

StagingBufferPrefetcher::StagingBufferPrefetcher(char* staging_buffer, unsigned long long int buffer_size, int node_id, int no_threads,
                                                 Sampler* sampler, StorageBackend* backend, PrefetcherBackend** pf_backends,
                                                 MetadataStore* metadata_store, DistributedManager* distr_manager,
                                                 TransformPipeline** transform_pipeline, int transform_output_size, Metrics* metrics,
                                                 bool collate_data) {
    this->buffer_size = buffer_size;
    this->staging_buffer = staging_buffer;
    this->node_id = node_id;
    this->no_threads = no_threads;
    this->sampler = new Sampler(*sampler);
    this->backend = backend;
    this->pf_backends = pf_backends;
    this->metadata_store = metadata_store;
    this->distr_manager = distr_manager;
    this->transform_pipeline = transform_pipeline;
    this->metrics = metrics;
    if (transform_pipeline != nullptr || collate_data) {
        batch_size = sampler->get_node_local_batch_size();
        unsigned long max_file_size = 0;
        for (int i = 0; i < backend->get_length(); i++) {
            unsigned long size = backend->get_file_size(i);
            int label_size = backend->get_label_size(i) + 1;
            if (size > max_file_size) {
                max_file_size = size;
            }
            if (label_size > largest_label_size) {
                largest_label_size = label_size;
            }
        }
        if (transform_pipeline != nullptr) {
          transform_buffers = new char*[no_threads];
          for (int i = 0; i < no_threads; i++) {
            transform_buffers[i] = new char[max_file_size];
          }
          this->transform_output_size = transform_output_size;
        }
    }
    this->collate_data = collate_data;
    global_iter_done = new bool[no_threads]();
}

StagingBufferPrefetcher::~StagingBufferPrefetcher() {
    delete sampler;
    delete[] global_iter_done;
    if (transform_pipeline != nullptr) {
        for (int i = 0; i < no_threads; i++) {
            delete[] transform_buffers[i];
        }
        delete[] transform_buffers;
    }
}

void StagingBufferPrefetcher::prefetch(int thread_id) {
    while (true) {
        std::vector<int> curr_access_string;
        sampler->get_node_access_string(node_id, curr_access_string);
        int access_string_size = curr_access_string.size();
        int inserted_until = 0;
        bool do_transform = transform_pipeline != nullptr;
        bool profiling = metrics != nullptr;
        std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
        while (true) {
            std::unique_lock<std::mutex> crit_section_lock(prefetcher_mutex);
            while (waiting_for_consumption) {
                consumption_waiting_cond_var.wait(crit_section_lock);
            }
            int j = prefetch_offset;
            if (j == 0) {
                curr_iter_file_ends.resize(access_string_size);
                curr_iter_file_ends_ready.resize(access_string_size);
                for (int i = 0; i < access_string_size; i++) {
                    curr_iter_file_ends_ready[i] = false;
                }
            }
            prefetch_offset += 1;

            if (j >= access_string_size) {
                break;
            }
            int file_id = curr_access_string[j];
            unsigned long file_size = backend->get_file_size(file_id);
            int label_size = backend->get_label_size(file_id);
            unsigned long entry_size = file_size + label_size + 1;
            if (do_transform) {
                // Batch mode, i.e. we fetch batch_size consecutive labels / samples
                entry_size = transform_output_size + label_size + 1;
                if (j % batch_size == 0) {
                    // If drop_last is false, can have smaller batches
                    curr_batch_size = std::min(access_string_size - j, batch_size);
                    // We're starting a new batch, need to check if there is enough space
                    while (staging_buffer_pointer < read_offset && staging_buffer_pointer + curr_batch_size * (transform_output_size + largest_label_size) >= read_offset) {
                        // Prevent overwriting of non-read data
                        waiting_for_consumption = true;
                        read_offset_cond_var.wait(crit_section_lock);
                    }
                }
            } else if (collate_data) {
              // Batch mode without transforming.
              if (j % batch_size == 0) {
                curr_batch_size = std::min(access_string_size - j, batch_size);
                while (staging_buffer_pointer < read_offset
                       && staging_buffer_pointer + curr_batch_size*(file_size + largest_label_size) >= read_offset) {
                  waiting_for_consumption = true;
                  read_offset_cond_var.wait(crit_section_lock);
                }
              }
            } else {
                while (staging_buffer_pointer < read_offset && staging_buffer_pointer + entry_size >= read_offset) {
                    // Prevent overwriting of non-read data
                    waiting_for_consumption = true;
                    read_offset_cond_var.wait(crit_section_lock);
                }
            }

            unsigned long long int local_staging_buffer_pointer;
            int batch_offset = 0;
            if (do_transform) {
                if (j % batch_size == batch_size - 1 || j == access_string_size - 1) {
                    if (staging_buffer_pointer + (curr_batch_size + batch_size) * (transform_output_size + largest_label_size) > buffer_size) {
                        staging_buffer_pointer = 0;
                        while (batch_size * (transform_output_size + largest_label_size) >= read_offset) {
                            waiting_for_consumption = true;
                            read_offset_cond_var.wait(crit_section_lock);
                        }
                    }

                    local_staging_buffer_pointer = staging_buffer_pointer;
                    staging_buffer_pointer += curr_batch_size * (transform_output_size + largest_label_size);
                } else {
                    local_staging_buffer_pointer = staging_buffer_pointer;
                }
                batch_offset = j % batch_size;
            } else if (collate_data) {
              if (j % batch_size == batch_size - 1
                  || j == access_string_size - 1) {
                if (staging_buffer_pointer + (curr_batch_size+batch_size)*(file_size+largest_label_size) > buffer_size) {
                  staging_buffer_pointer = 0;
                  while (batch_size * (file_size + largest_label_size) >= read_offset) {
                    waiting_for_consumption = true;
                    read_offset_cond_var.wait(crit_section_lock);
                  }
                }
                local_staging_buffer_pointer = staging_buffer_pointer;
                staging_buffer_pointer += curr_batch_size * (file_size+largest_label_size);
              } else {
                local_staging_buffer_pointer = staging_buffer_pointer;
              }
              batch_offset = j % batch_size;
            } else {
                if (staging_buffer_pointer + entry_size > buffer_size) {
                    // Start again at beginning of array
                    staging_buffer_pointer = 0;
                    // Ensure that overwriting is not possible after reset of pointer
                    while (entry_size >= read_offset) {
                        waiting_for_consumption = true;
                        read_offset_cond_var.wait(crit_section_lock);
                    }
                }
                local_staging_buffer_pointer = staging_buffer_pointer;
                staging_buffer_pointer += entry_size;
            }
            int curr_local_batch_size = curr_batch_size;

            if (waiting_for_consumption) {
                waiting_for_consumption = false;
                consumption_waiting_cond_var.notify_all();
            }
            crit_section_lock.unlock();

            backend->fetch_label(file_id, staging_buffer + local_staging_buffer_pointer + batch_offset * largest_label_size);
            if (do_transform || collate_data) {
                // Fill remaining bytes with zero bytes
                for (unsigned long long k = local_staging_buffer_pointer + batch_offset * largest_label_size + label_size + 1;
                    k < local_staging_buffer_pointer + (batch_offset + 1) * largest_label_size; k++) {
                    staging_buffer[k] = 0;
                }
            }
            if (!do_transform) {
              if (collate_data) {
                fetch(file_id,
                      staging_buffer + local_staging_buffer_pointer + curr_local_batch_size*largest_label_size + batch_offset*file_size,
                      thread_id);
              } else {
                fetch(file_id, staging_buffer + local_staging_buffer_pointer + label_size + 1, thread_id);
              }
            } else {
                fetch(file_id, transform_buffers[thread_id], thread_id);
                if (profiling) {
                    t1 = std::chrono::high_resolution_clock::now();
                }
                transform_pipeline[thread_id]->transform(transform_buffers[thread_id], file_size,
                        staging_buffer + local_staging_buffer_pointer + curr_local_batch_size * largest_label_size + batch_offset * transform_output_size);
                if (profiling) {
                    t2 = std::chrono::high_resolution_clock::now();
                    metrics->augmentation_time[thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count());
                }
            }
            std::unique_lock<std::mutex> staging_buffer_lock(staging_buffer_mutex);
            // Check if all the previous file ends were inserted to the queue. If not, don't insert, but only set
            // curr_iter_file_ends / curr_iter_file_ends_ready s.t. another thread will insert it
            if (!do_transform) {
              if (collate_data) {
                curr_iter_file_ends[j] = local_staging_buffer_pointer + curr_local_batch_size*largest_label_size + (batch_offset + 1) * file_size;
              } else {
                curr_iter_file_ends[j] = local_staging_buffer_pointer + entry_size;
              }
            } else {
                curr_iter_file_ends[j] = local_staging_buffer_pointer + curr_local_batch_size * largest_label_size + (batch_offset + 1) * transform_output_size;
            }
            curr_iter_file_ends_ready[j] = true;
            bool all_prev_inserted = true;
            for (int k = inserted_until; k < j; k++) {
                if (!curr_iter_file_ends_ready[k]) {
                    all_prev_inserted = false;
                    break;
                } else {
                    inserted_until = k;
                }
            }
            if (all_prev_inserted) {
                // Also insert file_ends from faster threads
                int k = j;
                bool inserted = false;
                while (k < access_string_size && curr_iter_file_ends_ready[k]) {
                    if ((!do_transform && !collate_data) || k % batch_size == batch_size - 1 || k == access_string_size - 1) {
                        file_ends.push_back(curr_iter_file_ends[k]);
                        inserted = true;
                    }
                    k++;
                }
                if (inserted) {
                    staging_buffer_cond_var.notify_one();
                }
            }
            staging_buffer_lock.unlock();
        }
        bool all_threads_done = true;

        // Advance batch when all threads are done with the current one
        std::unique_lock<std::mutex> crit_section_lock(prefetcher_mutex);
        global_iter_done[thread_id] = true;
        for (int i = 0; i < no_threads; i++) {
            if (!global_iter_done[i]) {
                all_threads_done = false;
            }
        }
        if (all_threads_done) {
            sampler->advance_batch();
            prefetch_batch += 1;
            prefetch_offset = 0;
            for (int i = 0; i < no_threads; i++) {
                global_iter_done[i] = false;
            }
            batch_advancement_cond_var.notify_all();
        } else {
            batch_advancement_cond_var.wait(crit_section_lock);
        }

        if (prefetch_batch >= sampler->epochs) {
            break;
        }

        crit_section_lock.unlock();
    }
}

void StagingBufferPrefetcher::fetch(int file_id, char* dst, int thread_id) {
    std::chrono::time_point<std::chrono::high_resolution_clock> t1, t2;
    bool profiling = metrics != nullptr;
    int remote_storage_level = distr_manager->get_remote_storage_class(file_id);
    int local_storage_level = metadata_store->get_storage_level(file_id);
    int option_order[3];
    metadata_store->get_option_order(local_storage_level, remote_storage_level, option_order);
    if (profiling) {
        t1 = std::chrono::high_resolution_clock::now();
    }
    if (option_order[0] == OPTION_REMOTE) {
        if (distr_manager->fetch(file_id, dst, thread_id)) {
            if (profiling) {
                t2 = std::chrono::high_resolution_clock::now();
                metrics->read_locations[0][thread_id].emplace_back(OPTION_REMOTE);
                metrics->read_times[0][thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count());
            }
            return;
        } else if (profiling) {
            // Track unsuccesful remote fetches as well
            metrics->read_locations[0][thread_id].emplace_back(-1);
            metrics->read_times[0][thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count());
        }
    }
    if (option_order[0] == OPTION_LOCAL || (option_order[0] == OPTION_REMOTE && option_order[1] == OPTION_LOCAL)) {
        pf_backends[local_storage_level - 1]->fetch(file_id, dst);
        if (profiling) {
            t2 = std::chrono::high_resolution_clock::now();
            metrics->read_locations[0][thread_id].emplace_back(OPTION_LOCAL);
            metrics->read_times[0][thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count());
        }
    } else {
        int planned_storage_level = metadata_store->get_planned_storage_level(file_id);
        if (planned_storage_level != -1) {
          // File is meant to be cached, but we are ahead of the prefetcher.
          // Use the current thread to help out.
          pf_backends[planned_storage_level - 1]->fetch_and_cache(file_id, dst);
        } else {
          backend->fetch(file_id, dst);
        }
        if (profiling) {
          // File is uncached, so both options hit the PFS.
          t2 = std::chrono::high_resolution_clock::now();
          metrics->read_locations[0][thread_id].emplace_back(OPTION_PFS);
          metrics->read_times[0][thread_id].emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count());
        }
    }
}

void StagingBufferPrefetcher::advance_read_offset(unsigned long long int new_offset) {
    std::unique_lock<std::mutex> lock(prefetcher_mutex);
    read_offset = new_offset;
    read_offset_cond_var.notify_one();
}

unsigned long long int StagingBufferPrefetcher::get_next_file_end() {
    std::unique_lock<std::mutex> staging_buffer_lock(staging_buffer_mutex);
    while (file_ends.empty()) {
        staging_buffer_cond_var.wait(staging_buffer_lock);
    }
    unsigned long long int file_end = file_ends.front();
    file_ends.pop_front();
    return file_end;
}
