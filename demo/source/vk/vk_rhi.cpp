#define VMA_IMPLEMENTATION
#include <string>
#include <iostream>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <map>
#include "vk_rhi.h"
#include "vk_helpers.h"

namespace hrr
{
    VulkanRHIConfig::VulkanRHIConfig(RHIConfig &config) :

        // Device extensions that will always be available
        device_extensions {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
            VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            //VK_EXT_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_EXTENSION_NAME
        },

        // Instance extensions that will always be available
        instance_extensions {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
            VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        },

        // By default, there are no validation layers
        validation_layers{},

        // Device features
        device_features {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = nullptr,
            .features = {},
        },
        device_feature_timeline_semaphore {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            .pNext = nullptr,
            .timelineSemaphore = VK_TRUE,
        },
        device_feature_acceleration_structure {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
            .pNext = nullptr,
            .accelerationStructure = VK_TRUE,
            .accelerationStructureCaptureReplay = VK_FALSE,
            .accelerationStructureIndirectBuild = VK_FALSE,
            .accelerationStructureHostCommands = VK_FALSE,
            .descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE,
        },
        device_features_ray_tracing {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
            .pNext = nullptr,
            .rayTracingPipeline = VK_TRUE,
            .rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE,
            .rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE,
            .rayTracingPipelineTraceRaysIndirect = VK_FALSE,
            .rayTraversalPrimitiveCulling = VK_FALSE,
        },
        device_features_buffer_device_address {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT,
            .pNext = nullptr,
            .bufferDeviceAddress = VK_TRUE,
            .bufferDeviceAddressCaptureReplay = VK_FALSE,
            .bufferDeviceAddressMultiDevice = VK_FALSE,
        },
        device_features_dynamic_rendering {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
            .pNext = nullptr,
            .dynamicRendering = VK_TRUE,
        },
        device_features_synchronization2 {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
            .pNext = nullptr,
            .synchronization2 = VK_TRUE,
        }
    {
        void** device_feature_next = &device_features.pNext;
        const auto add_device_feature = [&device_feature_next](auto& feature_data) {
            *device_feature_next = &feature_data;
            device_feature_next = &feature_data.pNext;
        };

        // Features that we will always want:
        add_device_feature(device_features_synchronization2);
        //add_device_feature(device_feature_timeline_semaphore);
        //add_device_feature(device_features_buffer_device_address);
        //add_device_feature(device_features_dynamic_rendering);

        if (config.raytracing)
        {
            // Raytracing specific device extensions
            device_extensions.insert(device_extensions.end(), {
                VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, // required by VK_KHR_acceleration_structure
                VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
            });

            // Raytracing specific device features
            add_device_feature(device_feature_acceleration_structure);
            add_device_feature(device_features_ray_tracing);
        }

        if (config.debug)
        {
            validation_layers.push_back("VK_LAYER_KHRONOS_validation");
        }
    }

    VulkanRHI::VulkanRHI(RHIConfig &config, GLFWwindow* window)
        : RHI(config, window)
        , config(config)
        , instance(nullptr)
        , surface(nullptr)
        , device(nullptr)
        , device_gpu(nullptr)
        , device_queue(nullptr)
        , device_queue_family_index(-1)
        , command_pool(nullptr)
        , command_buffer(nullptr)
        , swapchain(nullptr)
        , swapchain_images{}
        , swapchain_views{}
        , semaphores{.image_acquired=nullptr, .render_complete=nullptr}
    {
        setup();
    }

    VulkanRHI::~VulkanRHI()
    {
        shutdown();
    }

    RHIProgram* VulkanRHI::create_shader(RHIProgram::Type type, const std::string& name, const std::vector<uint8_t>& bytes)
    {
        log << "Creating VK shader module from " << bytes.size() << " bytes... ";
        VkShaderModuleCreateInfo shader_module_create_info
        {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = bytes.size(),
            .pCode = reinterpret_cast<const uint32_t*>(bytes.data()),
        };

        VkShaderModule shader_module;
        if (!vk_check(vkCreateShaderModule(device, &shader_module_create_info, nullptr, &shader_module)))
        {
            return nullptr;
        }

        static const std::map<RHIProgram::Type, VkShaderStageFlagBits> shader_type_map
        {
            { RHIProgram::Type::Vertex, VK_SHADER_STAGE_VERTEX_BIT },
            { RHIProgram::Type::Geometry, VK_SHADER_STAGE_GEOMETRY_BIT },
            { RHIProgram::Type::Compute, VK_SHADER_STAGE_COMPUTE_BIT },
            { RHIProgram::Type::Fragment, VK_SHADER_STAGE_FRAGMENT_BIT },
        };

        VkPipelineShaderStageCreateInfo stage_info
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = shader_type_map.at(type),
            .module = shader_module,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        };

        log << "done\n";

        // Track the shader module internally so it can be deleted at the right time,
        // and return a ptr to the program.
        programs.emplace_back(std::make_unique<VulkanRHIProgram>(name, shader_module, stage_info));
        return programs.back().get();
    }

    RHIPipeline* VulkanRHI::create_pipeline(const std::vector<RHIProgram*>& stages)
    {
        VulkanRHIPipeline rhi_pipeline;

        {
            log << "Creating VK pipeline layout... ";


            VkPipelineLayoutCreateInfo pipeline_layout_create_info
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .setLayoutCount = 0,
                .pSetLayouts = nullptr,
                .pushConstantRangeCount = 0,
                .pPushConstantRanges = nullptr,
            };

            rhi_pipeline.layout = nullptr;
            if (vk_check(vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &rhi_pipeline.layout)))
            {
                log << "done\n";
            }
        }

        {
            log << "Creating VK render pass... ";

            VkAttachmentDescription color_attachment
            {
                .flags = 0,
                .format = swapchain_format.format,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            };

            VkAttachmentReference attachment_reference
            {
                .attachment = 0,
                .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            };

            VkSubpassDescription subpass_description
            {
                .flags = 0,
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .inputAttachmentCount = 0,
                .pInputAttachments = nullptr,
                .colorAttachmentCount = 1,
                .pColorAttachments = &attachment_reference,
                .pResolveAttachments = nullptr,
                .pDepthStencilAttachment = nullptr,
                .preserveAttachmentCount = 0,
                .pPreserveAttachments = nullptr,
            };

            VkSubpassDependency subpass_dependency
            {
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dependencyFlags = 0,
            };

            VkRenderPassCreateInfo render_pass_create_info
            {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .attachmentCount = 1,
                .pAttachments = &color_attachment,
                .subpassCount = 1,
                .pSubpasses = &subpass_description,
                .dependencyCount = 1,
                .pDependencies = &subpass_dependency,
            };


            rhi_pipeline.render_pass = nullptr;
            if (vk_check(vkCreateRenderPass(device, &render_pass_create_info, nullptr, &rhi_pipeline.render_pass)))
            {
                log << "done\n";
            }
        }

        {
            log << "Creating VK pipeline... ";

            VkPipelineVertexInputStateCreateInfo pipeline_vertex_input_state_create_info
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .vertexBindingDescriptionCount = 0,
                .pVertexBindingDescriptions = nullptr, // optional
                .vertexAttributeDescriptionCount = 0,
                .pVertexAttributeDescriptions = nullptr, // optional
            };

            VkPipelineInputAssemblyStateCreateInfo pipeline_input_assembly_state_create_info
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .primitiveRestartEnable = VK_FALSE,
            };

            VkViewport viewport
            {
                .x = 0.0f,
                .y = 0.0f,
                .width = (float)swapchain_extent.width,
                .height = (float)swapchain_extent.height,
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };

            VkRect2D scissor
            {
                .offset = {0, 0},
                .extent = swapchain_extent,
            };

            /* ????
            std::vector<VkDynamicState> dynamic_states = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo pipeline_dynamic_state_create_info
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
                .pDynamicStates = dynamic_states.data(),
            };
            */

            VkPipelineViewportStateCreateInfo pipeline_viewport_state_create_info
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .viewportCount = 1,
                .pViewports = &viewport,
                .scissorCount = 1,
                .pScissors = &scissor,
            };

            VkPipelineRasterizationStateCreateInfo pipeline_rasterization_state_create_info
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_NONE,
                .frontFace = VK_FRONT_FACE_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
                .depthBiasConstantFactor = 0.0f, // optional
                .depthBiasClamp = 0.0f, // optional
                .depthBiasSlopeFactor = 0.0f,
                .lineWidth = 1.0f,
            };

            VkPipelineMultisampleStateCreateInfo pipeline_multisample_state_create_info
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = VK_FALSE,
                .minSampleShading = 1.0f, // optional
                .pSampleMask = nullptr, // optional
                .alphaToCoverageEnable = VK_FALSE, // optional
                .alphaToOneEnable = VK_FALSE, // optional
            };

            VkPipelineColorBlendAttachmentState pipeline_color_blend_attachment_state
            {
                //
                // TODO: Modify this if we want alpha blending
                //
                .blendEnable = VK_FALSE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                .colorBlendOp = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                .alphaBlendOp = VK_BLEND_OP_ADD,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            };

            VkPipelineColorBlendStateCreateInfo pipeline_color_blend_state_create_info
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = VK_FALSE,
                .logicOp = VK_LOGIC_OP_COPY,
                .attachmentCount = 1,
                .pAttachments = &pipeline_color_blend_attachment_state,
                .blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f },
            };

            std::vector<VkPipelineShaderStageCreateInfo> shader_stage_create_infos;
            shader_stage_create_infos.resize(stages.size());
            for (size_t i = 0; i < stages.size(); ++i)
            {
                auto program = dynamic_cast<VulkanRHIProgram*>(stages[i]);
                shader_stage_create_infos[i] = program->stage_info;
            }

            VkGraphicsPipelineCreateInfo graphics_pipeline_create_info
            {
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = (uint32_t)shader_stage_create_infos.size(),
                .pStages = shader_stage_create_infos.data(),
                .pVertexInputState = &pipeline_vertex_input_state_create_info,
                .pInputAssemblyState = &pipeline_input_assembly_state_create_info,
                .pViewportState = &pipeline_viewport_state_create_info,
                .pRasterizationState = &pipeline_rasterization_state_create_info,
                .pMultisampleState = &pipeline_multisample_state_create_info,
                .pDepthStencilState = nullptr,
                .pColorBlendState = &pipeline_color_blend_state_create_info,
                .pDynamicState = nullptr,//&pipeline_dynamic_state_create_info,
                .layout = rhi_pipeline.layout,
                .renderPass = rhi_pipeline.render_pass,
                .subpass = 0,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = -1,
            };

            if (vk_check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphics_pipeline_create_info, nullptr, &rhi_pipeline.pipeline)))
            {
                log << "done\n";
            }
        }

        {
            log << "Creating pipeline framebuffers for " << swapchain_views.size() << " swapchain views... ";
            rhi_pipeline.framebuffers.resize(swapchain_views.size());
            bool success = true;
            for (size_t i = 0; i < swapchain_views.size(); ++i)
            {
                VkImageView attachments[] = { swapchain_views[i] };
                VkFramebufferCreateInfo frame_buffer_create_info
                {
                    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    .renderPass = rhi_pipeline.render_pass,
                    .attachmentCount = 1,
                    .pAttachments = attachments,
                    .width = swapchain_extent.width,
                    .height = swapchain_extent.height,
                    .layers = 1,
                };
                success &= vk_check(vkCreateFramebuffer(device, &frame_buffer_create_info, nullptr, &rhi_pipeline.framebuffers[i]));
            }
            if (success)
            {
                log << "done\n";
            }
        }

        pipelines.emplace_back(std::make_unique<VulkanRHIPipeline>(rhi_pipeline));
        return pipelines.back().get();
    }

    void VulkanRHI::dispatch(RHIPipeline* pipeline)
    {
        auto vk_pipeline = dynamic_cast<VulkanRHIPipeline*>(pipeline);

        // Don't start until previous frame is done
        vkWaitForFences(device, 1, &frame_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &frame_fence);

        // Begin render pass
        uint32_t image_index;
        vk_check(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, semaphores.image_acquired, VK_NULL_HANDLE, &image_index));

        // Clear the command buffer
        vkResetCommandBuffer(command_buffer, 0);

        {
            // Begin command buffer
            VkCommandBufferBeginInfo command_buffer_begin_info
            {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = 0,
                .pInheritanceInfo = nullptr,
            };
            vk_check(vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info));
        }

        {
            VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
            VkRenderPassBeginInfo render_pass_begin_info
            {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .pNext = nullptr,
                .renderPass = vk_pipeline->render_pass,
                .framebuffer = vk_pipeline->framebuffers[image_index],
                .renderArea = { .offset = {0, 0}, .extent = swapchain_extent },
                .clearValueCount = 1,
                .pClearValues = &clear_color,
            };

            vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
        }

        {
            // Execute pipeline
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline->pipeline);

            /*
            VkViewport viewport
            {
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(swapchain_extent.width),
                .height = static_cast<float>(swapchain_extent.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);

            VkRect2D scissor
            {
                .offset = {0, 0},
                .extent = swapchain_extent,
            };
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);
            */

            vkCmdDraw(command_buffer, 3, 1, 0, 0);
        }

        // Stop Recording command buffer
        vkCmdEndRenderPass(command_buffer);
        vk_check(vkEndCommandBuffer(command_buffer));

        VkSemaphore wait_semaphores[] = { semaphores.image_acquired };
        VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSemaphore signal_semaphores[] = { semaphores.render_complete };
        {
            // Submit commands
            VkSubmitInfo submit_info
            {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = wait_semaphores,
                .pWaitDstStageMask = wait_stages,
                .commandBufferCount = 1,
                .pCommandBuffers = &command_buffer,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = signal_semaphores,
            };
            vk_check(vkQueueSubmit(device_queue, 1, &submit_info, frame_fence));
        }

        VkSwapchainKHR swapchains[] = { swapchain };
        {
            VkPresentInfoKHR present_info
            {
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .waitSemaphoreCount = 1,
                .pWaitSemaphores = signal_semaphores,
                .swapchainCount = 1,
                .pSwapchains = swapchains,
                .pImageIndices = &image_index,
                .pResults = nullptr,
            };

            vk_check(vkQueuePresentKHR(device_queue, &present_info));
        }
    }

    bool VulkanRHI::vk_check(VkResult result)
    {
        if (result < 0)
        {
            //err << std::string("Error: ") << string_VkResult(result) << std::endl;
            err << std::string("Error: ") << result << std::endl;
            return false;
        }

        return true;
    }

    void VulkanRHI::setup()
    {
        // Start up Volk
        volkInitialize();

        create_instance();
        create_surface();
        create_device();
        create_commands();
        create_swapchain();
        create_synchronization();
    }

    void VulkanRHI::shutdown()
    {
        destroy_pipelines();
        destroy_shaders();
        destroy_synchronization();
        destroy_swapchain();
        destroy_commands();
        destroy_device();
        destroy_surface();
        destroy_instance();

        // Not necessary (according to volk docs)
        volkFinalize();
    }


    void VulkanRHI::create_instance()
    {
        if (instance)
        {
            destroy_instance();
        }

        // Check validation layers
        {
            // Build array of available layers
            uint32_t num_instance_layers;
            vkEnumerateInstanceLayerProperties(&num_instance_layers, nullptr);
            std::vector<VkLayerProperties> instance_layers(num_instance_layers);
            vkEnumerateInstanceLayerProperties(&num_instance_layers, instance_layers.data());

            // Make sure we have all layers
            log << "Checking VK validation layers...\n";
            bool all_layers_found = true;
            for (const char *required_layer: config.validation_layers)
            {
                bool layer_found = false;
                for (const auto &instance_layer: instance_layers)
                {
                    if (strcmp(required_layer, instance_layer.layerName) == 0)
                    {
                        layer_found = true;
                        break;
                    }
                }

                all_layers_found &= layer_found;
                log << " - " << required_layer << ": " << (layer_found ? "found" : "missing") << "\n";
            }

            if (!all_layers_found)
            {
                err << "Not all required validations layers found\n";
            }
        }

        // Check instance extensions
        {
            uint32_t count = 0;
            vk_check(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
            std::vector<VkExtensionProperties> extension_properties(count);
            vk_check(vkEnumerateInstanceExtensionProperties(nullptr, &count, extension_properties.data()));

            log << "Checking VK instance extensions...\n";
            bool all_supported = true;
            for (auto name: config.instance_extensions)
            {
                bool supported = false;
                for (const auto &property: extension_properties)
                {
                    if (!strcmp(property.extensionName, name))
                    {
                        supported = true;
                        break;
                    }
                }

                all_supported &= supported;
                log << " - " << name << ": " << (supported ? "supported" : "unsupported") << "\n";
            }

            if (!all_supported)
            {
                err << "Some required instance extensions are not available!\n";
            }
        }

        VkApplicationInfo app_info
        {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "hardrare",
            .applicationVersion = 0,
            .pEngineName = "",
            .engineVersion = 0,
            .apiVersion = VK_API_VERSION_1_3
        };

        VkInstanceCreateInfo instance_create_info
        {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = (uint32_t) config.validation_layers.size(),
            .ppEnabledLayerNames = config.validation_layers.data(),
            .enabledExtensionCount = (uint32_t) config.instance_extensions.size(),
            .ppEnabledExtensionNames = config.instance_extensions.data(),
        };

        vk_check(vkCreateInstance(&instance_create_info, nullptr, &instance));

        // Load instance specific vulkan function ptrs
        volkLoadInstance(instance);
    }

    void VulkanRHI::destroy_instance()
    {
        if (instance)
        {
            log << "Destroying VK instance...\n";
            vkDestroyInstance(instance, nullptr);
            instance = nullptr;
        }
    }

    void VulkanRHI::create_surface()
    {
        if (surface)
        {
            destroy_surface();
        }

        vk_check(glfwCreateWindowSurface(instance, window, nullptr, &surface));
    }

    void VulkanRHI::destroy_surface()
    {
        if (surface)
        {
            log << "Destroying VK surface...\n";
            vkDestroySurfaceKHR(instance, surface, nullptr);
            surface = nullptr;
        }
    }

    void VulkanRHI::create_device()
    {
        if (device)
        {
            destroy_device();
        }

        // select physical device
        {
            log << "Looking for VK physical device...\n";

            uint32_t gpu_count = 0;
            vk_check(vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr));
            if (gpu_count == 0)
            {
                err << "There are no Vulkan physical devices available\n";
            }

            //if (params.physical_device_index >= (int)gpu_count) {
            //    vk.error(std::format("Requested physical device index (zero-based) is {}, but physical device count is {}",
            //        params.physical_device_index, gpu_count));
            //}
            std::vector<VkPhysicalDevice> physical_devices(gpu_count);
            vk_check(vkEnumeratePhysicalDevices(instance, &gpu_count, physical_devices.data()));

            int selected_gpu = -1;
            VkPhysicalDeviceProperties gpu_properties{};

            for (const auto& physical_device : physical_devices)
            {
                vkGetPhysicalDeviceProperties(physical_device, &gpu_properties);

                log << " - " << gpu_properties.deviceName;
                if (selected_gpu == -1 &&
                    VK_VERSION_MAJOR(gpu_properties.apiVersion) == 1 &&
                    VK_VERSION_MINOR(gpu_properties.apiVersion) >= 3)
                {
                    selected_gpu = int(&physical_device - physical_devices.data());
                    log << " [SELECTED]";
                }
                log << "\n";
            }

            if (selected_gpu == -1)
            {
                err << "Failed to select physical device that supports Vulkan 1.3 or higher\n";
            }

            device_gpu = physical_devices[selected_gpu];
        }

        // select queue family
        {
            log << "Looking for VK queue family which supports graphics and presentation...\n";

            uint32_t queue_family_count;
            vkGetPhysicalDeviceQueueFamilyProperties(device_gpu, &queue_family_count, nullptr);

            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(device_gpu, &queue_family_count, queue_families.data());

            // select queue family with presentation and graphics support
            device_queue_family_index = -1;
            for (uint32_t i = 0; i < queue_family_count; i++)
            {
                VkBool32 presentation_supported;
                vk_check(vkGetPhysicalDeviceSurfaceSupportKHR(device_gpu, i, surface, &presentation_supported));

                log << " - Queue family " << i << " ";

                if (presentation_supported && (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
                {
                    log << "adequate";
                    if (device_queue_family_index == -1)
                    {
                        device_queue_family_index = i;
                        log << " [SELECTED]";
                    }
                }
                else
                {
                    log << "inadequate";
                }

                log << "\n";
            }

            if (device_queue_family_index == uint32_t(-1))
            {
                err << "Failed to find queue family\n";
            }
        }

        // create VkDevice
        {
            log << "Checking VK device for extensions...\n";

            uint32_t count = 0;
            vk_check(vkEnumerateDeviceExtensionProperties(device_gpu, nullptr, &count, nullptr));
            std::vector<VkExtensionProperties> extension_properties(count);
            vk_check(vkEnumerateDeviceExtensionProperties(device_gpu, nullptr, &count, extension_properties.data()));

            auto is_extension_supported = [&extension_properties](const char* extension_name)
            {
                for (const auto& property : extension_properties)
                {
                    if (!strcmp(property.extensionName, extension_name))
                    {
                        return true;
                    }
                }
                return false;
            };

            bool all_supported = true;
            for (auto required_extension : config.device_extensions)
            {
                bool supported = is_extension_supported(required_extension);
                all_supported &= supported;
                log << " - " << required_extension << ": " << (supported ? "supported" : "UNSUPPORTED") << "\n";
            }

            if (!all_supported)
            {
                err << "Device does not support all required extensions!\n";
            }
        }

        const float priority = 1.0;
        VkDeviceQueueCreateInfo queue_create_info
        {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .queueFamilyIndex = device_queue_family_index,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        };

        VkDeviceCreateInfo device_create_info
        {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &config.device_features,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_create_info,
            .enabledExtensionCount = (uint32_t)config.device_extensions.size(),
            .ppEnabledExtensionNames = config.device_extensions.data(),
            .pEnabledFeatures = nullptr,
        };

        vk_check(vkCreateDevice(device_gpu, &device_create_info, nullptr, &device));

        // Load device specific vulkan function ptrs
        volkLoadDevice(device);

        // Previously we analyzed the devices queue families and selected one based on
        // its properties. Here we'll get a reference to the queue itself which we'll
        // use for queueing up commands.
        vkGetDeviceQueue(device, device_queue_family_index, 0, &device_queue);
    }

    void VulkanRHI::destroy_device()
    {
        if (device)
        {
            log << "Destroying device... ";
            vkDeviceWaitIdle(device);
            vkDestroyDevice(device, nullptr);
            device = nullptr;
            device_gpu = nullptr;
            device_queue = nullptr;
            log << "done\n";
        }
    }

    void VulkanRHI::create_commands()
    {
        {
            log << "Creating command pool... ";

            VkCommandPoolCreateInfo command_pool_create_info
            {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = device_queue_family_index,
            };

            vk_check(vkCreateCommandPool(device, &command_pool_create_info, nullptr, &command_pool));

            log << "done\n";
        }

        {
            log << "Creating command buffer... ";

            VkCommandBufferAllocateInfo command_buffer_allocate_info
            {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = command_pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };

            vk_check(vkAllocateCommandBuffers(device, &command_buffer_allocate_info, &command_buffer));

            log << " done\n";
        }
    }

    void VulkanRHI::destroy_commands()
    {
        if (command_pool)
        {
            log << "Destroying command buffer... ";
            vkDeviceWaitIdle(device);
            vkDestroyCommandPool(device, command_pool, nullptr);
            command_pool = nullptr;
            command_buffer = nullptr;
            log << "done\n";
        }
    }

    void VulkanRHI::create_swapchain()
    {
        VkSurfaceCapabilitiesKHR supported_capabilities;
        std::vector<VkSurfaceFormatKHR> supported_formats;
        std::vector<VkPresentModeKHR> supported_presentation_modes;
        {
            log << "Finding supported swapchain details... ";

            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_gpu, surface, &supported_capabilities);

            uint32_t supported_format_count;
            vkGetPhysicalDeviceSurfaceFormatsKHR(device_gpu, surface, &supported_format_count, nullptr);
            if (supported_format_count != 0)
            {
                supported_formats.resize(supported_format_count);
                vkGetPhysicalDeviceSurfaceFormatsKHR(device_gpu, surface, &supported_format_count, supported_formats.data());
            }
            else
            {
                err << "No surface formats supported\n";
                return;
            }

            uint32_t presentation_mode_count;
            vkGetPhysicalDeviceSurfacePresentModesKHR(device_gpu, surface, &presentation_mode_count, nullptr);
            if (presentation_mode_count != 0)
            {
                supported_presentation_modes.resize(presentation_mode_count);
                vkGetPhysicalDeviceSurfacePresentModesKHR(device_gpu, surface, &presentation_mode_count, supported_presentation_modes.data());
            }
            else
            {
                err << "No surface presentation modes supported\n";
                return;
            }

            log << "done\n";
        }

        // Surface format
        log << "Choosing swapchain surface format...\n";
        bool format_found = false;
        swapchain_format = supported_formats[0];
        for (const VkSurfaceFormatKHR& supported_format : supported_formats)
        {
            //log << " - " << string_VkFormat(supported_format.format)
            //    << " - " << string_VkColorSpaceKHR(supported_format.colorSpace);
            log << " - " << (supported_format.format)
                << " - " << (supported_format.colorSpace);

            // NOTE: Need to figure out how to determine what's good here...
            if (supported_format.format == VK_FORMAT_B8G8R8A8_UNORM &&
                supported_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                swapchain_format = supported_format;
                format_found = true;
                log << " [SELECTED]";
            }

            log << "\n";
        }
        if (!format_found)
        {
            err << "No adequate format was found! Reverting to first option, but this may not work.\n";
        }

        // Presentation modes
        log << "Choosing swapchain presentation mode...";
        VkPresentModeKHR presentation_mode = VK_PRESENT_MODE_FIFO_KHR;
        for (const VkPresentModeKHR& supported_presentation_mode : supported_presentation_modes)
        {
            if (supported_presentation_mode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                presentation_mode = supported_presentation_mode;
                break;
            }
        }
        log << "done\n";

        // Extents
        log << "Choosing swapchain extents...";
        swapchain_extent = supported_capabilities.currentExtent;
        if (supported_capabilities.currentExtent.width == (uint32_t)std::numeric_limits<uint32_t>::max())
        {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            swapchain_extent.width = std::clamp((uint32_t)width, supported_capabilities.minImageExtent.width, supported_capabilities.maxImageExtent.width);
            swapchain_extent.height = std::clamp((uint32_t)height, supported_capabilities.minImageExtent.height, supported_capabilities.maxImageExtent.height);
        }
        log << " done. Chose " << swapchain_extent.width << " x " << swapchain_extent.height << "\n";

        // Image count
        log << "Choosing swapchain image count...";
        uint32_t image_count = std::clamp((uint32_t)supported_capabilities.minImageCount, (uint32_t)0, (uint32_t)supported_capabilities.maxImageCount);
        log << " done. Chose " << image_count << "\n";

        {
            log << "Creating swapchain...";

            // Create swapchain data
            VkSwapchainCreateInfoKHR swapchain_create_info
            {
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .pNext = nullptr,
                .flags = 0,
                .surface = surface,
                .minImageCount = image_count,
                .imageFormat = swapchain_format.format,
                .imageColorSpace = swapchain_format.colorSpace,
                .imageExtent = swapchain_extent,

                // Specifies the amount of layers each image consists of.
                // Always 1 unless you are developing a stereoscopic 3D application
                // (https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain)
                .imageArrayLayers = 1,
                .imageUsage =
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,// |
                    //VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    //VK_IMAGE_USAGE_STORAGE_BIT,

                // NOTE: For now we have just one queue, so... exclusive???
                .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 1, // optional?
                .pQueueFamilyIndices = &device_queue_family_index, // optional?

                // NOTE: Use currentTransform to indicate that you do not want an
                // additional transform to be applied to the image
                .preTransform = supported_capabilities.currentTransform,

                // Specifies if the alpha channel should be used for blending with other windows
                // in the window system.
                // (https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain)
                .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,

                //
                .presentMode = presentation_mode,

                // If the clipped member is set to VK_TRUE then that means that we don't care
                // about the color of pixels that are obscured, for example because another
                // window is in front of them.
                // (https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain)
                .clipped = VK_TRUE,
                .oldSwapchain = VK_NULL_HANDLE,
            };

            vk_check(vkCreateSwapchainKHR(device, &swapchain_create_info, nullptr, &swapchain));

            log << " done\n";
        }

        {
            log << "Tracking swapchain images...";
            swapchain_images.resize(image_count);
            vk_check(vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.data()));
            log << " done\n";
        }

        {
            log << "Creating swapchain image views...";
            swapchain_views.resize(image_count);
            for (size_t i = 0; i < image_count; ++i)
            {
                VkImageViewCreateInfo image_view_create_info
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .image = swapchain_images[i],
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = swapchain_format.format,
                    .components = {
                        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                    },
                    .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    }
                };

                vk_check(vkCreateImageView(device, &image_view_create_info, nullptr, &swapchain_views[i]));
            }
            log << " done\n";
        }
    }

    void VulkanRHI::destroy_swapchain()
    {
        if (!swapchain_views.empty())
        {
            log << "Destroying " << swapchain_views.size() << " swapchain image views...\n";
            for (VkImageView& view : swapchain_views)
            {
                vkDeviceWaitIdle(device);
                vkDestroyImageView(device, view, nullptr);
            }
        }

        if (swapchain)
        {
            log << "Destroying swapchain...\n";
            vkDeviceWaitIdle(device);
            vkDestroySwapchainKHR(device, swapchain, nullptr);
            swapchain = nullptr;
        }
    }

    /*
    void VulkanRHI::create_framebuffers()
    {
    }

    void VulkanRHI::destroy_framebuffers()
    {
    }
    */

    void VulkanRHI::create_synchronization()
    {
        log << "Creating synchronization structures... ";

        VkSemaphoreCreateInfo semaphore_create_info
        {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        };

        vk_check(vkCreateSemaphore(device, &semaphore_create_info, nullptr, &semaphores.image_acquired));
        vk_check(vkCreateSemaphore(device, &semaphore_create_info, nullptr, &semaphores.render_complete));

        VkFenceCreateInfo fence_create_info
        {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        vk_check(vkCreateFence(device, &fence_create_info, nullptr, &frame_fence));

        log << "done\n";
    }

    void VulkanRHI::destroy_synchronization()
    {
        log << "Destroying synchronization structures... ";
        if (semaphores.image_acquired)
        {
            vkDeviceWaitIdle(device);
            vkDestroySemaphore(device, semaphores.image_acquired, nullptr);
            semaphores.image_acquired = nullptr;
        }

        if (semaphores.render_complete)
        {
            vkDeviceWaitIdle(device);
            vkDestroySemaphore(device, semaphores.render_complete, nullptr);
            semaphores.render_complete = nullptr;
        }

        if (frame_fence)
        {
            vkDeviceWaitIdle(device);
            vkDestroyFence(device, frame_fence, nullptr);
            frame_fence = nullptr;
        }
        log << "done\n";
    }

    void VulkanRHI::destroy_shaders()
    {
        if (!programs.empty())
        {
            log << "Destroying " << programs.size() << " shader modules... ";
            for (std::unique_ptr<VulkanRHIProgram>& program : programs)
            {
                vkDeviceWaitIdle(device);
                vkDestroyShaderModule(device, program->shader_module, nullptr);
            }
            programs.clear();
            log << "done\n";
        }
    }

    void VulkanRHI::destroy_pipelines()
    {
        if (!pipelines.empty())
        {
            log << "Destroying " << pipelines.size() << " pipelines... ";
            for (std::unique_ptr<VulkanRHIPipeline>& pipeline : pipelines)
            {
                vkDeviceWaitIdle(device);

                for (VkFramebuffer& framebuffer : pipeline->framebuffers)
                {
                    vkDestroyFramebuffer(device, framebuffer, nullptr);
                }

                if (pipeline->pipeline)
                {
                    vkDestroyPipeline(device, pipeline->pipeline, nullptr);
                }

                if (pipeline->render_pass)
                {
                    vkDestroyRenderPass(device, pipeline->render_pass, nullptr);
                }

                if (pipeline->layout)
                {
                    vkDestroyPipelineLayout(device, pipeline->layout, nullptr);
                }
            }
            log << "done\n";
        }
    }
}