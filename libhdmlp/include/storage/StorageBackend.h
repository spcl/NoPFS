#ifndef HDMLP_STORAGEBACKEND_H
#define HDMLP_STORAGEBACKEND_H


#include <string>
#include <unordered_map>

class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    virtual int get_length() = 0;

    virtual void fetch_label(int file_id, char* dst) = 0;

    virtual int get_label_size(int file_id) = 0;

    virtual unsigned long get_file_size(int file_id) = 0;

    virtual void fetch(int file_id, char* dst) = 0;
};


#endif //HDMLP_STORAGEBACKEND_H
