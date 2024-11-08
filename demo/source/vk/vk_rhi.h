#pragma once
#include <vector>
#include <tuple>
#include <memory>
#include "vk_helpers.h"
#include "../rhi.h"

namespace hrr
{
    struct VulkanRHIConfig
    {
        VulkanRHIConfig(RHIConfig &config);

        std::vector<const char *> device_extensions;
        std::vector<const char *> instance_extensions;
        std::vector<const char *> validation_layers;

        // Device features
        VkPhysicalDeviceFeatures2 device_features;
        VkPhysicalDeviceTimelineSemaphoreFeatures device_feature_timeline_semaphore;
        VkPhysicalDeviceAccelerationStructureFeaturesKHR device_feature_acceleration_structure;
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR device_features_ray_tracing;
        VkPhysicalDeviceBufferDeviceAddressFeatures device_features_buffer_device_address;
        VkPhysicalDeviceDynamicRenderingFeatures device_features_dynamic_rendering;
        VkPhysicalDeviceSynchronization2Features device_features_synchronization2;
    };

    struct VulkanRHIProgram : public RHIProgram
    {
        explicit VulkanRHIProgram(
            const std::string& name,
            VkShaderModule shader_module,
            VkPipelineShaderStageCreateInfo stage_info)
            : RHIProgram(name)
            , shader_module(shader_module)
            , stage_info(stage_info) { }

        ~VulkanRHIProgram() override = default;

        VkShaderModule shader_module;
        VkPipelineShaderStageCreateInfo stage_info;
    };

    struct VulkanRHIPipeline : public RHIPipeline
    {
        VkPipelineLayout layout;
        VkRenderPass render_pass;
        VkPipeline pipeline;
        std::vector<VkFramebuffer> framebuffers;
    };

    class VulkanRHI : public RHI
    {
    public:
        explicit VulkanRHI(RHIConfig &config, GLFWwindow* window);

        ~VulkanRHI() override;

        RHIProgram* create_shader(RHIProgram::Type type, const std::string& name, const std::vector<uint8_t>& bytes) override;

        RHIPipeline* create_pipeline(const std::vector<RHIProgram*>& stages) override;

        void dispatch(RHIPipeline* pipeline) override;

    private:

        bool vk_check(VkResult result);

        void setup();
        void shutdown();

        void create_instance();
        void destroy_instance();

        void create_surface();
        void destroy_surface();

        void create_device();
        void destroy_device();

        void create_commands();
        void destroy_commands();

        void create_swapchain();
        void destroy_swapchain();

        void create_synchronization();
        void destroy_synchronization();

        void destroy_shaders();
        void destroy_pipelines();

        const VulkanRHIConfig config;

        VkInstance instance;
        VkSurfaceKHR surface;
        VkDevice device;
        VkPhysicalDevice device_gpu;
        VkQueue device_queue;
        uint32_t device_queue_family_index;
        VkCommandPool command_pool;
        VkCommandBuffer command_buffer;

        // Shaders created by this RHI
        std::vector<std::unique_ptr<VulkanRHIProgram>> programs;
        std::vector<std::unique_ptr<VulkanRHIPipeline>> pipelines;

        // Swapchain
        VkSwapchainKHR swapchain;
        VkExtent2D swapchain_extent;
        VkSurfaceFormatKHR swapchain_format;
        std::vector<VkImage> swapchain_images;
        std::vector<VkImageView> swapchain_views;

        // Synchronization semaphores
        struct {
            VkSemaphore image_acquired;     // Swap chain image presentation
            VkSemaphore render_complete;    // Swap chain image presentation
        } semaphores;
        VkFence frame_fence;

        VkRenderPass render_pass;



        /*
        uint32_t                        queue_family_index;
        VkDevice                        device;
        VkQueue                         queue;
        double                          timestamp_period_ms;

        VmaAllocator                    allocator;

        */


        /*
        VkImageUsageFlags               surface_usage_flags;
        VkSurfaceFormatKHR              surface_format;
        VkExtent2D                      surface_size;
        Swapchain_Info                  swapchain_info;

        uint32_t                        swapchain_image_index = -1; // current swapchain image

        VkCommandPool                   command_pools[2];
        VkCommandBuffer                 command_buffers[2];
        VkCommandBuffer                 command_buffer; // command_buffers[frame_index]
        int                             frame_index;

        VkSemaphore                     image_acquired_semaphore[2];
        VkSemaphore                     rendering_finished_semaphore[2];
        VkFence                         frame_fence[2];

        VkQueryPool                     timestamp_query_pools[2];
        VkQueryPool                     timestamp_query_pool; // timestamp_query_pool[frame_index]
        uint32_t                        timestamp_query_count;

        // Host visible memory used to copy image data to device local memory.
        VkBuffer                        staging_buffer;
        VmaAllocation                   staging_buffer_allocation;
        VkDeviceSize                    staging_buffer_size;
        uint8_t*                        staging_buffer_ptr; // pointer to mapped staging buffer

        VkDebugUtilsMessengerEXT        debug_utils_messenger;

        VkDescriptorPool                imgui_descriptor_pool;
        */
    };
}
