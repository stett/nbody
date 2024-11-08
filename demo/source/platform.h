#pragma once
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace hrr
{
    struct PlatformConfig
    {
        std::ostream &log = std::cout;
        std::ostream &err = std::cout;
        std::string resource_root = ".";
    };

    class Platform
    {
    public:
        explicit Platform(const PlatformConfig& config)
            : log(config.log)
            , err(config.err)
        {
            set_resource_root(config.resource_root);
        }
        virtual ~Platform() = default;

        void set_resource_root(const std::string& _resource_root)
        {
            log << "Setting platform resource root to "
                << std::filesystem::absolute(_resource_root)
                << "\n";
            resource_root = _resource_root;
        }

        std::filesystem::path get_resource_path(const std::string& relative_path)
        {
            return (std::filesystem::path(resource_root) / relative_path);
        }

        std::vector<uint8_t> get_resource_bytes(const std::string& relative_path)
        {
            log << "Getting bytes from file " << relative_path << "... ";

            std::filesystem::path file_path = get_resource_path(relative_path);

            std::ifstream file(file_path, std::ios_base::in | std::ios_base::binary);
            if (!file)
            {
                err << "failed to open file: " << file_path << "\n";
                return {};
            }

            // get file size
            file.seekg(0, std::ios_base::end);
            std::streampos file_size = file.tellg();
            file.seekg(0, std::ios_base::beg);
            if (file_size == std::streampos(-1) || !file)
            {
                err << "failed to read file stats: " << file_path << "\n";
                return {};
            }

            // read file content
            std::vector<uint8_t> file_content(static_cast<size_t>(file_size));
            file.read(reinterpret_cast<char*>(file_content.data()), file_size);
            if (!file)
            {
                err << "failed to read file content: " << file_path << "\n";
            }

            log << "done. Read " << file_size << " bytes.\n";

            return file_content;
        }


    private:
        std::ostream &log;
        std::ostream &err;
        std::string resource_root;
    };
}
