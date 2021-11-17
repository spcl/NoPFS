#include "../../include/storage/FileSystemBackend.h"
#include <dirent.h>
#include <codecvt>
#include <locale>
#include <algorithm>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string.h>


FileSystemBackend::FileSystemBackend(const std::wstring& path, bool checkpoint, std::string checkpoint_path, int node_id) {
    using type = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<type, wchar_t> converter;
    std::string str_path = converter.to_bytes(path);
    if (str_path == "env") {
      char* env_path = std::getenv("HDMLP_FILESYSTEM_PATH");
      if (env_path == nullptr) {
        throw std::runtime_error("HDMLP_FILESYSTEM_PATH not set");
      }
      str_path = env_path;
    }
    if (str_path.back() != '/') {
        str_path += '/';
    }
    this->path = str_path;
    this->checkpoint = checkpoint;
    if (checkpoint) {
        if (checkpoint_path == "") {
            this->checkpoint_path = this->path;
        } else {
            this->checkpoint_path = checkpoint_path;
        }
        this->init_mappings_from_checkpoint(node_id);
    } else {
        this->init_mappings(node_id);
    }
}

int FileSystemBackend::get_length() {
    return id_mappings.size();
}

void FileSystemBackend::init_mappings(int node_id) {
    std::vector<FileInformation> file_information;
    struct dirent* entry = nullptr;
    DIR* dp = nullptr;

    dp = opendir(path.c_str());
    if (dp == nullptr) {
        throw std::runtime_error("Invalid path specified");
    }
    while ((entry = readdir(dp))) {
        if (entry->d_name[0] != '.') {
            std::string dir_name = entry->d_name;
            struct dirent* subentry = nullptr;
            DIR* subp = nullptr;

            subp = opendir((path + dir_name).c_str());
            if (subp == nullptr) {
                // Not a directory
                continue;
            }

            while ((subentry = readdir(subp))) {
                std::string rel_path = entry->d_name;
                rel_path += '/';
                rel_path += subentry->d_name;
                std::string file_name = abs_path(&rel_path);
                int fd = open(file_name.c_str(), O_RDONLY);
                struct stat stbuf; // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
                fstat(fd, &stbuf);
                close(fd);
                if (!S_ISREG(stbuf.st_mode)) { // NOLINT(hicpp-signed-bitwise)
                    // Not a regular file
                    continue;
                }
                struct FileInformation fi{};
                fi.label = entry->d_name;
                fi.file_name = subentry->d_name;
                fi.file_size = stbuf.st_size;
                file_information.push_back(fi);
            }

            closedir(subp);


        }
    }
    closedir(dp);
    // Ensure that all nodes have same file ids by sorting them
    std::sort(file_information.begin(), file_information.end(), [](FileInformation& a, FileInformation& b) {
                  return a.label + a.file_name > b.label + b.file_name;
              }
    );
    std::ofstream checkpoint_stream;
    if (checkpoint && node_id == 0) {
        checkpoint_stream.open(checkpoint_path + "/hdmlp_metadata_",  std::ofstream::out | std::ofstream::trunc);
    }
    for (const auto& fi : file_information) {
        id_mappings.push_back(fi.file_name);
        label_mappings.push_back(fi.label);
        size_mappings.push_back(fi.file_size);
        if (checkpoint && node_id == 0) {
            checkpoint_stream << fi.file_name << "," << fi.label << "," << fi.file_size << std::endl;
        }
    }
    if (checkpoint && node_id == 0) {
        checkpoint_stream.close();
        // Rename after file was completely written to ensure no nodes see partially written files:
        std::rename((checkpoint_path + "/hdmlp_metadata_").c_str(), (checkpoint_path + "/hdmlp_metadata").c_str());
    }
}


void FileSystemBackend::init_mappings_from_checkpoint(int node_id) {
    std::ifstream checkpoint_stream(checkpoint_path + "/hdmlp_metadata");
    if (checkpoint_stream.fail()) {
        this->init_mappings(node_id);
    }
    std::string line;
    unsigned long file_id = 0;
    while(std::getline(checkpoint_stream, line)) {
        std::stringstream ss(line);

        std::string label;
        std::string file_name;
        int file_size;
        std::getline(ss, file_name, ',');
        std::getline(ss, label, ',');
        ss >> file_size;
        id_mappings.push_back(file_name);
        label_mappings.push_back(label);
        size_mappings.push_back(file_size);

        file_id++;
    }

}

std::string FileSystemBackend::abs_path(const std::string* rel_path) {
    return path + *rel_path;
}

unsigned long FileSystemBackend::get_file_size(int file_id) {
    return size_mappings[file_id];
}

/**
 * Fetch the file from the backend
 *
 * @param file_id
 * @param dst
 */
void FileSystemBackend::fetch(int file_id, char* dst) {
    std::string label = label_mappings[file_id];
    std::string rel_path = label + '/' + id_mappings[file_id];
    std::string file_name = abs_path(&rel_path);
    unsigned long entry_size = get_file_size(file_id);
    FILE* f = fopen(file_name.c_str(), "rb");
    fread(dst, 1, entry_size, f);
    fclose(f);
}

int FileSystemBackend::get_label_size(int file_id) {
    return label_mappings[file_id].size();
}

void FileSystemBackend::fetch_label(int file_id, char* dst) {
    strcpy(dst, label_mappings[file_id].c_str());
}

