#ifndef HDMLP_PREFETCHERBACKEND_H
#define HDMLP_PREFETCHERBACKEND_H

class PrefetcherBackend {
public:
    virtual ~PrefetcherBackend() = default;

    virtual void prefetch(int thread_id, int storage_class) = 0;

    virtual void fetch(int file_id, char* dst) = 0;

    virtual char* get_location(int file_id, unsigned long* len) = 0;

    virtual int get_prefetch_offset() = 0;

    virtual bool is_done() = 0;

  virtual void fetch_and_cache(int file_id, char* dst) {
    throw std::runtime_error("Not implemented");
  }
};

#endif //HDMLP_PREFETCHERBACKEND_H
