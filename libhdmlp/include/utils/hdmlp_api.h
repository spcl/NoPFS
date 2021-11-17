#ifndef HDMLP_HDMLP_API_H
#define HDMLP_HDMLP_API_H
#define PARALLEL_JOBS_LIMIT 255
extern "C" {
Prefetcher* pf[PARALLEL_JOBS_LIMIT];
bool used_map[PARALLEL_JOBS_LIMIT] = {false};
unsigned long long int consumed_until[PARALLEL_JOBS_LIMIT] = {0};

int setup(wchar_t* dataset_path,
          wchar_t* config_path,
          int batch_size,
          int epochs,
          int distr_scheme,
          bool drop_last_batch,
          int seed,
          wchar_t** transform_names,
          char* transform_args,
          int transform_output_size,
          int transform_len,
          wchar_t* filesystem_backend,
          wchar_t* hdf5_data_name,
          wchar_t* hdf5_target_name,
          bool collate_data);

char* get_staging_buffer(int job_id);

int get_node_id(int job_id);

int get_no_nodes(int job_id);

int length(int job_id);

unsigned long long int get_next_file_end(int job_id);

void destroy(int job_id);

int get_metric_size(int job_id, wchar_t* kind, int index, int subindex);

double* get_stall_time(int job_id);

double* get_augmentation_time(int job_id, int thread_id);

double* get_read_times(int job_id, int storage_class, int thread_id);

int* get_read_locations(int job_id, int storage_class, int thread_id);

int get_label_distance(int job_id);
};
#endif //HDMLP_HDMLP_API_H
