#ifndef HDMLP_FILESYSTEMBACKEND_H
#define HDMLP_FILESYSTEMBACKEND_H


#include <vector>
#include "StorageBackend.h"

class FileSystemBackend : public StorageBackend {
public:
    FileSystemBackend(const std::wstring& path, bool checkpoint, std::string checkpoint_path, int node_id);

    void fetch_label(int file_id, char* dst) override;

    virtual int get_label_size(int file_id) override;

    int get_length() override;

    unsigned long get_file_size(int file_id) override;

    void fetch(int file_id, char* dst) override;

private:
    struct FileInformation {
        std::string label;
        std::string file_name;
        int file_size;
    };
    std::string path;
    std::vector<std::string> label_mappings;
    std::vector<int> size_mappings;
    std::vector<std::string> id_mappings;
    bool checkpoint;
    std::string checkpoint_path;

    void init_mappings(int node_id);

    std::string abs_path(const std::string* rel_path);

    void init_mappings_from_checkpoint(int node_id);
};


#endif //HDMLP_FILESYSTEMBACKEND_H
