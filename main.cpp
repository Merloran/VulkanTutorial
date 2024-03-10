#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#undef max
#undef min
#include <vulkan/vulkan.h>
#include <iostream>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <random>
#include <tiny_obj_loader.h>

#include "Types.hpp"

constexpr UInt32 WIDTH = 800;
constexpr UInt32 HEIGHT = 600;
constexpr Int32 MAX_FRAMES_IN_FLIGHT = 2;
constexpr UInt32 PARTICLE_COUNT = 16384;

const std::vector<const Char*> validationLayers = 
{
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const Char *> deviceExtensions = 
{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
constexpr Bool ENABLE_VALIDATION_LAYERS = false;
#else
constexpr Bool ENABLE_VALIDATION_LAYERS = true;
#endif

struct QueueFamilyIndices
{
    std::optional<UInt32> graphicsAndComputeFamily;
    std::optional<UInt32> presentFamily;

    Bool is_complete()
	{
        return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
    }
};

struct Vertex
{
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 uv;

    static VkVertexInputBindingDescription s_get_binding_description()
	{
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding   = 0;
        bindingDescription.stride    = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> s_get_attribute_descriptions()
	{
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        //Position
        attributeDescriptions[0].binding  = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset   = offsetof(Vertex, position);
        //Color
        attributeDescriptions[1].binding  = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset   = offsetof(Vertex, color);
        //UV
        attributeDescriptions[2].binding  = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format   = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset   = offsetof(Vertex, uv);

        return attributeDescriptions;
    }

    Bool operator==(const Vertex &other) const
	{
        return position == other.position && color == other.color && uv == other.uv;
    }
};

namespace std
{
    template<>
	struct hash<Vertex>
	{
        size_t operator()(Vertex const &vertex) const
    	{
            return ((hash<glm::vec3>()(vertex.position) ^
                    (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                (hash<glm::vec2>()(vertex.uv) << 1);
        }
    };
}

struct UniformBufferObject
{
    Float32 deltaTime;
    //glm::mat4 model;
    //glm::mat4 view;
    //glm::mat4 projection;
};

struct Particle
{
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec4 color;


    static VkVertexInputBindingDescription s_get_binding_description()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding   = 0;
        bindingDescription.stride    = sizeof(Particle);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> s_get_attribute_descriptions()
	{
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding  = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format   = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset   = offsetof(Particle, position);

        attributeDescriptions[1].binding  = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset   = offsetof(Particle, color);

        return attributeDescriptions;
    }
};

struct SwapChainSupportDetails
{
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
    VkSurfaceCapabilitiesKHR capabilities;
};

class HelloTriangleApplication
{
public:
    Void run()
	{
        init_window();
        init_vulkan();
        update();
        shutdown();
    }

private:
    #pragma region Variables
    const std::string GLSL_COMPILER_PATH = "D:/VulkanSDK/Bin/glslc.exe";
    const std::string SHADERS_PATH = "Shaders/";
    const std::string TEXTURES_PATH = "Textures/";
    const std::string MODELS_PATH = "Models/";
    std::vector<std::string> shaderFiles = 
    {
        "shader.vert",
        "shader.frag",
        "shader.comp"
    };
    GLFWwindow *window;
	VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice = nullptr;
    VkDevice logicalDevice;

    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue computeQueue;

    VkSwapchainKHR           swapChain;
    std::vector<VkImage>     swapChainImages;
    VkFormat                 swapChainImageFormat;
    VkExtent2D               swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;

    VkRenderPass renderPass;

    VkDescriptorSetLayout        descriptorSetLayout;
    VkDescriptorPool             descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    VkPipelineLayout graphicsPipelineLayout;
    VkPipeline       graphicsPipeline;

    VkPipelineLayout computePipelineLayout;
    VkPipeline       computePipeline;

    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkCommandPool commandPool;

    VkImage        depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView    depthImageView;

    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    VkImage        colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView    colorImageView;

    UInt32         mipLevels;
    VkImage        textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView    textureImageView;
    VkSampler      textureSampler;

    std::vector<Vertex> vertices;
    std::vector<UInt32> indices;

    VkBuffer       vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    VkBuffer       indexBuffer;
    VkDeviceMemory indexBufferMemory;

    std::vector<VkBuffer>       uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<Void *>         uniformBuffersMapped;

    std::vector<VkBuffer>       shaderStorageBuffers;
    std::vector<VkDeviceMemory> shaderStorageBuffersMemory;

    std::vector<VkCommandBuffer> graphicsCommandBuffers;
    std::vector<VkCommandBuffer> computeCommandBuffers;

    std::vector<VkFence>     computeInFlightFences;
    std::vector<VkSemaphore> computeFinishedSemaphores;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    UInt32 currentFrame = 0;
    Float32 lastFrameTime = 0.0f;
    Bool framebufferResized = false;
	#pragma endregion Variables

    // Methods are arranged in call order, but methods like s_debug_callback is before shutdown
    // because it could be not called and this is something like helper method and shutdown is 
    // last method and every method used in shutdown will be in inverted order over shutdown
    Void init_window()
	{
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "VulkanTutorial", nullptr, nullptr);
        glfwSetFramebufferSizeCallback(window, s_framebuffer_resize_callback);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, s_framebuffer_resize_callback);
    }

    Void init_vulkan()
	{
        create_instance();
        setup_debug_messenger();
        create_surface();
        pick_physical_device();
        create_logical_device();
        create_swap_chain();
        create_image_views();
        create_render_pass();
        create_descriptor_set_layout();
        create_graphics_pipeline();
        create_compute_pipeline();
        create_command_pool();
        create_color_resources();
        create_depth_resources();
        create_framebuffers();
        create_texture_image();
        create_texture_image_view();
        create_texture_sampler();
        //load_model();
        //create_vertex_buffer();
        //create_index_buffer();
        create_uniform_buffers();
        create_shader_storage_buffers();
        create_descriptor_pool();
        create_descriptor_sets();
        create_command_buffers();
        create_compute_command_buffers();
        create_synchronization_objects();
    }

    Void create_instance()
	{
        if (ENABLE_VALIDATION_LAYERS && !check_validation_layer_support()) 
        {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        //APP INFO
        VkApplicationInfo appInfo{};
        appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName   = "Vulkan Tutorial";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0U, 1U, 0U, 0U);
        appInfo.pEngineName        = "RayEngine";
        appInfo.engineVersion      = VK_MAKE_API_VERSION(0U, 1U, 0U, 0U);
        appInfo.apiVersion         = VK_API_VERSION_1_0;


        //INSTANCE CREATE INFO
        VkInstanceCreateInfo createInfo{};
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        std::vector<const Char*> extensions = get_required_extensions();

        createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo        = &appInfo;
        if (ENABLE_VALIDATION_LAYERS) 
        {
            populate_debug_messenger_create_info(debugCreateInfo);
            createInfo.enabledLayerCount   = static_cast<UInt32>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
            createInfo.pNext               = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
        } else {
            createInfo.enabledLayerCount   = 0U;
            createInfo.pNext               = nullptr;
        }
        createInfo.enabledExtensionCount   = static_cast<UInt32>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        //TODO: in the far future think about using custom allocator
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create instance!");
        }
    }

    Bool check_validation_layer_support()
    {
        UInt32 layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const Char *layerName : validationLayers)
        {
            Bool layerFound = false;

            for (const VkLayerProperties &layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
            {
                return false;
            }
        }

        return true;
    }

    std::vector<const Char *> get_required_extensions()
    {
        UInt32 glfwExtensionCount = 0U;
        const Char **glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const Char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (ENABLE_VALIDATION_LAYERS)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    Void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = s_debug_callback;
    }

    Void setup_debug_messenger()
    {
        if (!ENABLE_VALIDATION_LAYERS)
        {
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populate_debug_messenger_create_info(createInfo);

        if (create_debug_utils_messenger_ext(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to set up debug messenger!");
        }

    }

    VkResult create_debug_utils_messenger_ext(VkInstance instance,
                                              const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                              const VkAllocationCallbacks *pAllocator,
                                              VkDebugUtilsMessengerEXT *pDebugMessenger)
    {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr)
        {
            return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        } else {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }

    Void create_surface()
	{
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create window surface!");
        }
    }

	Void pick_physical_device()
	{
        // Check for device with Vulkan support
        UInt32 deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) 
        {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        // Check for device that is suitable
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        for (const VkPhysicalDevice &device : devices) 
        {
            if (is_device_suitable(device)) 
            {
                physicalDevice = device;
                msaaSamples = get_max_usable_sample_count();
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    Bool is_device_suitable(VkPhysicalDevice device)
	{
        QueueFamilyIndices indices = find_queue_families(device);

        const Bool areExtensionsSupported = check_device_extension_support(device);

        Bool isSwapChainAdequate = false;
        if (areExtensionsSupported) 
        {
            SwapChainSupportDetails swapChainSupport = query_swap_chain_support(device);
            isSwapChainAdequate = !swapChainSupport.formats.empty()
        					   && !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return indices.is_complete() &&
        	   areExtensionsSupported && 
        	   isSwapChainAdequate && 
               supportedFeatures.samplerAnisotropy;
    }

    QueueFamilyIndices find_queue_families(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices;
        UInt32 queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        for (Int32 i = 0; i < queueFamilies.size(); ++i)
        {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT && 
                queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                indices.graphicsAndComputeFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport)
            {
                indices.presentFamily = i;
            }

            if (indices.is_complete())
            {
                break;
            }
        }

        return indices;
    }

    Bool check_device_extension_support(VkPhysicalDevice device)
	{
        UInt32 extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const VkExtensionProperties &extension : availableExtensions) 
        {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    SwapChainSupportDetails query_swap_chain_support(VkPhysicalDevice device)
    {
        SwapChainSupportDetails details;
        UInt32 formatCount;
        UInt32 presentModeCount;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device,
                                                 surface,
                                                 &formatCount,
                                                 details.formats.data());
        }

        vkGetPhysicalDeviceSurfacePresentModesKHR(device,
                                                  surface,
                                                  &presentModeCount,
                                                  nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device,
                                                      surface,
                                                      &presentModeCount,
                                                      details.presentModes.data());
        }

        return details;
    }

    VkSampleCountFlagBits get_max_usable_sample_count()
	{
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

        VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts
    							  & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
        if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
        if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
        if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
        if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
        if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

        return VK_SAMPLE_COUNT_1_BIT;
    }

    Void create_logical_device()
    {
        QueueFamilyIndices indices = find_queue_families(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<UInt32> uniqueQueueFamilies = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

        Float32 queuePriority = 1.0f;
        for (UInt32 queueFamily : uniqueQueueFamilies) 
        {
            //QUEUE CREATE INFO
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount       = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        //PHYSICAL DEVICE FEATURES
        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        deviceFeatures.sampleRateShading = VK_TRUE; // enable sample shading feature for the device

        //DEVICE CREATE INFO
        VkDeviceCreateInfo createInfo{};
        createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount    = static_cast<UInt32>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos       = queueCreateInfos.data();
        createInfo.pEnabledFeatures        = &deviceFeatures;
        createInfo.enabledExtensionCount   = static_cast<UInt32>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        if (ENABLE_VALIDATION_LAYERS)
        {
            createInfo.enabledLayerCount   = static_cast<UInt32>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount   = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(logicalDevice, indices.presentFamily.value(), 0, &presentQueue);
        vkGetDeviceQueue(logicalDevice, indices.graphicsAndComputeFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(logicalDevice, indices.graphicsAndComputeFamily.value(), 0, &computeQueue);
    }

    Void create_swap_chain()
	{
        SwapChainSupportDetails swapChainSupport = query_swap_chain_support(physicalDevice);
        VkSurfaceFormatKHR      surfaceFormat    = choose_swap_surface_format(swapChainSupport.formats);
        VkPresentModeKHR        presentMode      = choose_swap_present_mode(swapChainSupport.presentModes);
        VkExtent2D              extent           = choose_swap_extent(swapChainSupport.capabilities);
        UInt32                  imageCount       = swapChainSupport.capabilities.minImageCount + 1;

        if (swapChainSupport.capabilities.maxImageCount > 0) 
        {
            imageCount = std::min(imageCount, swapChainSupport.capabilities.maxImageCount);
        }

        VkSwapchainCreateInfoKHR createInfo{};
        QueueFamilyIndices indices  = find_queue_families(physicalDevice);
        UInt32 queueFamilyIndices[] = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

        createInfo.sType                     = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface                   = surface;
        createInfo.minImageCount             = imageCount;
        createInfo.imageFormat               = surfaceFormat.format;
        createInfo.imageColorSpace           = surfaceFormat.colorSpace;
        createInfo.imageExtent               = extent;
        createInfo.imageArrayLayers          = 1;
        createInfo.imageUsage                = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (indices.graphicsAndComputeFamily != indices.presentFamily) 
        {
            createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices   = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices   = nullptr; // Optional
        }
        createInfo.preTransform              = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha            = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode               = presentMode;
        createInfo.clipped                   = VK_TRUE;
        createInfo.oldSwapchain              = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create swap chain!");
        }
        vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, swapChainImages.data());
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent      = extent;
    }

    VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR> &availableFormats)
	{
        for (const VkSurfaceFormatKHR &availableFormat : availableFormats) 
        { //TODO: change it to RGBA instead of BGRA
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && 
                availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) 
            {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR> &availablePresentModes)
	{
        for (const VkPresentModeKHR &availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) 
            {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR &capabilities)
	{
        if (capabilities.currentExtent.width != std::numeric_limits<UInt32>::max())
        {
            return capabilities.currentExtent;
        } else {
            Int32 width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = 
            {
                static_cast<UInt32>(width),
                static_cast<UInt32>(height)
            };

            actualExtent.width  = std::clamp(actualExtent.width, 
                                             capabilities.minImageExtent.width, 
                                             capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height,
                                             capabilities.minImageExtent.height,
                                             capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

	Void create_image_views()
	{
        swapChainImageViews.resize(swapChainImages.size());

        for (UInt32 i = 0; i < swapChainImages.size(); ++i) 
        {
            swapChainImageViews[i] = create_image_view(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }

    Void create_render_pass()
	{
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format         = swapChainImageFormat;
        colorAttachment.samples        = msaaSamples;
        colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        
        
        VkAttachmentDescription colorAttachmentResolve{};
        colorAttachmentResolve.format         = swapChainImageFormat;
        colorAttachmentResolve.samples        = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentResolve.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        
        VkAttachmentReference colorAttachmentResolveRef{};
        colorAttachmentResolveRef.attachment = 1;
        colorAttachmentResolveRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


        VkAttachmentDescription depthAttachment{};
        depthAttachment.format         = find_depth_format();
        depthAttachment.samples        = msaaSamples;
        depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 2;
        depthAttachmentRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;
        subpass.pResolveAttachments     = &colorAttachmentResolveRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass    = 0;
        dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, colorAttachmentResolve, depthAttachment };
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<UInt32>(attachments.size());
        renderPassInfo.pAttachments    = attachments.data();
        renderPassInfo.subpassCount    = 1;
        renderPassInfo.pSubpasses      = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies   = &dependency;

        if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    Void create_descriptor_set_layout()
	{
        std::array<VkDescriptorSetLayoutBinding, 4> layoutBindings{};
        layoutBindings[0].binding            = 0;
        layoutBindings[0].descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        layoutBindings[0].descriptorCount    = 1;
        layoutBindings[0].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
        layoutBindings[0].pImmutableSamplers = nullptr; // Optional
        
        layoutBindings[1].binding            = 1;
        layoutBindings[1].descriptorCount    = 1;
        layoutBindings[1].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;
        
        layoutBindings[2].binding            = 2;
        layoutBindings[2].descriptorCount    = 1;
        layoutBindings[2].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
        
        layoutBindings[3].binding            = 3;
        layoutBindings[3].descriptorCount    = 1;
        layoutBindings[3].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[3].pImmutableSamplers = nullptr;
        layoutBindings[3].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<UInt32>(layoutBindings.size());
        layoutInfo.pBindings    = layoutBindings.data();

        if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    Void create_graphics_pipeline()
	{
        compile_shaders();
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        //pipelineLayoutInfo.setLayoutCount         = 1; // Optional
        //pipelineLayoutInfo.pSetLayouts            = &descriptorSetLayout; // Optional
        //pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        //pipelineLayoutInfo.pPushConstantRanges    = nullptr; // Optional

        if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &graphicsPipelineLayout) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create graphics pipeline layout!");
        }

        std::vector<Char> vertShaderCode = s_read_file(SHADERS_PATH + shaderFiles[0] + ".spv");
        std::vector<Char> fragShaderCode = s_read_file(SHADERS_PATH + shaderFiles[1] + ".spv");

        VkShaderModule vertShaderModule = create_shader_module(vertShaderCode);
        VkShaderModule fragShaderModule = create_shader_module(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName  = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName  = "main";

        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = { vertShaderStageInfo, fragShaderStageInfo };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        VkVertexInputBindingDescription bindingDescription = Particle::s_get_binding_description();
        auto attributeDescriptions = Particle::s_get_attribute_descriptions();
        vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount   = 1;
        vertexInputInfo.pVertexBindingDescriptions      = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<UInt32>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions    = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        std::vector<VkDynamicState> dynamicStates = 
        {
		    VK_DYNAMIC_STATE_VIEWPORT,
		    VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<UInt32>(dynamicStates.size());
        dynamicState.pDynamicStates    = dynamicStates.data();

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount  = 1;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable       = VK_TRUE;
        depthStencil.depthWriteEnable      = VK_TRUE;
        depthStencil.depthCompareOp        = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds        = 0.0f; // Optional
        depthStencil.maxDepthBounds        = 1.0f; // Optional
        depthStencil.stencilTestEnable     = VK_FALSE;
        depthStencil.front                 = {}; // Optional
        depthStencil.back                  = {}; // Optional

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable        = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth               = 1.0f;
        rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable         = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // Optional
        rasterizer.depthBiasClamp          = 0.0f; // Optional
        rasterizer.depthBiasSlopeFactor    = 0.0f; // Optional

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples  = msaaSamples;
        multisampling.sampleShadingEnable   = VK_TRUE; // enable sample shading in the pipeline
        multisampling.minSampleShading      = 0.2f; // min fraction for sample shading; closer to one is smooth
        multisampling.pSampleMask           = nullptr; // Optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampling.alphaToOneEnable      = VK_FALSE; // Optional

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT
    	                                         | VK_COLOR_COMPONENT_G_BIT
    	                                         | VK_COLOR_COMPONENT_B_BIT
    	                                         | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable         = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; // Optional
        colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD; // Optional

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable     = VK_FALSE;
        colorBlending.logicOp           = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount   = 1;
        colorBlending.pAttachments      = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount          = static_cast<UInt32>(shaderStages.size());
        pipelineInfo.pStages             = shaderStages.data();
        pipelineInfo.pVertexInputState   = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState   = &multisampling;
        pipelineInfo.pDepthStencilState  = &depthStencil;
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.pDynamicState       = &dynamicState;
        pipelineInfo.layout              = graphicsPipelineLayout;
        pipelineInfo.renderPass          = renderPass;
        pipelineInfo.subpass             = 0;
        pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex   = -1; // Optional

        if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
        vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
    }

    Void create_compute_pipeline()
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts    = &descriptorSetLayout;

        if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }
        
        std::vector<Char> compShaderCode   = s_read_file(SHADERS_PATH + shaderFiles[2] + ".spv");
        VkShaderModule    compShaderModule = create_shader_module(compShaderCode);

        VkPipelineShaderStageCreateInfo compShaderStageInfo{};
        compShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compShaderStageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        compShaderStageInfo.module = compShaderModule;
        compShaderStageInfo.pName  = "main";

    	VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage  = compShaderStageInfo;

        if (vkCreateComputePipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        vkDestroyShaderModule(logicalDevice, compShaderModule, nullptr);
    }

    Void compile_shaders()
    {
        std::string compileParameters = "-o";
        std::string resultExtension = ".spv";
        for (const std::string& shader : shaderFiles)
        {
            std::stringstream command;
            command << GLSL_COMPILER_PATH << " " << SHADERS_PATH << shader << " "
        			<< compileParameters  << " " << SHADERS_PATH << shader << resultExtension;
            const Int32 code = system(command.str().c_str());
            if (ENABLE_VALIDATION_LAYERS)
            {
                if (code == 0)
                {
                    SPDLOG_INFO("Compiling {} ended with code: {}", shader, code);
                } else {
                    SPDLOG_ERROR("Compiling {} ended with code: {}", shader, code);
                }
            }
        }
    }

    static std::vector<Char> s_read_file(const std::string &fileName)
	{
        std::ifstream file(fileName, std::ios::ate | std::ios::binary);

        if (!file.is_open()) 
        {
            throw std::runtime_error("failed to open file!" + fileName);
        }

        const UInt64 fileSize = (UInt64)file.tellg();
        std::vector<Char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    VkShaderModule create_shader_module(const std::vector<Char> &code)
	{
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode    = reinterpret_cast<const UInt32*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    Void create_framebuffers()
	{
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (UInt64 i = 0; i < swapChainImageViews.size(); ++i) 
        {
            // ORDER HERE MUST MATCH ORDER IN CREATE RENDER PASS ATTACHMENTS NUMBERS
            std::array<VkImageView, 3> attachments =
            {
            	colorImageView,
                swapChainImageViews[i],
                depthImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass      = renderPass;
            framebufferInfo.attachmentCount = static_cast<UInt32>(attachments.size());
            framebufferInfo.pAttachments    = attachments.data();
            framebufferInfo.width           = swapChainExtent.width;
            framebufferInfo.height          = swapChainExtent.height;
            framebufferInfo.layers          = 1;

            if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) 
            {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    Void create_command_pool()
	{
        QueueFamilyIndices queueFamilyIndices = find_queue_families(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();

        if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    Void create_color_resources()
	{
        VkFormat colorFormat = swapChainImageFormat;

        create_image(swapChainExtent.width, 
                     swapChainExtent.height, 
                     1, 
                     msaaSamples, 
                     colorFormat, 
                     VK_IMAGE_TILING_OPTIMAL, 
                     VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                     colorImage, 
                     colorImageMemory);
        colorImageView = create_image_view(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }

    Void create_depth_resources()
	{
        VkFormat depthFormat = find_depth_format();

        create_image(swapChainExtent.width, 
                     swapChainExtent.height, 
                     1,
                     msaaSamples,
                     depthFormat,
                     VK_IMAGE_TILING_OPTIMAL, 
                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                     depthImage, 
                     depthImageMemory);
        depthImageView = create_image_view(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
    }

    VkFormat find_depth_format()
	{
        return find_supported_format({
            					     VK_FORMAT_D32_SFLOAT,
            					     VK_FORMAT_D32_SFLOAT_S8_UINT,
            					     VK_FORMAT_D24_UNORM_S8_UINT
					                 },
					                 VK_IMAGE_TILING_OPTIMAL,
					                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    VkFormat find_supported_format(const std::vector<VkFormat> &candidates, 
                                   VkImageTiling tiling, 
                                   VkFormatFeatureFlags features)
	{
        for (VkFormat format : candidates) 
        {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) 
            {
                return format;
            }
        	else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) 
            {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    Bool has_stencil_component(VkFormat format)
    {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    Void create_texture_image()
	{
        Int32 texWidth, texHeight, texChannels;
        stbi_set_flip_vertically_on_load(true);
        stbi_uc *pixels = stbi_load((MODELS_PATH + "viking_room/viking_room.png").c_str(), 
                                    &texWidth, 
                                    &texHeight, 
                                    &texChannels, 
                                    STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) 
        {
            throw std::runtime_error("failed to load texture image!");
        }
        mipLevels = static_cast<UInt32>(glm::floor(std::log2(glm::max(texWidth, texHeight)))) + 1U;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        create_buffer(imageSize, 
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                      stagingBuffer, 
                      stagingBufferMemory);

        Void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, pixels, static_cast<UInt64>(imageSize));
        vkUnmapMemory(logicalDevice, stagingBufferMemory);
        stbi_image_free(pixels);

        create_image(texWidth, 
                     texHeight,
                     mipLevels,
                     VK_SAMPLE_COUNT_1_BIT,
                     VK_FORMAT_R8G8B8A8_SRGB, 
                     VK_IMAGE_TILING_OPTIMAL, 
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                     textureImage, 
                     textureImageMemory);

        transition_image_layout(textureImage, 
                                VK_FORMAT_R8G8B8A8_SRGB, 
                                VK_IMAGE_LAYOUT_UNDEFINED, 
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
                                mipLevels);

        copy_buffer_to_image(stagingBuffer, 
                             textureImage, 
                             static_cast<UInt32>(texWidth), 
                             static_cast<UInt32>(texHeight));

        generate_mipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    Void create_image(UInt32 width, 
                      UInt32 height,
                      UInt32 mipLevels, 
                      VkSampleCountFlagBits numSamples,
                      VkFormat format, 
                      VkImageTiling tiling, 
                      VkImageUsageFlags usage, 
                      VkMemoryPropertyFlags properties, 
                      VkImage &image, 
                      VkDeviceMemory &imageMemory)
    {
	    
        VkImageCreateInfo imageInfo{};
        imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType     = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width  = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth  = 1;
        imageInfo.mipLevels     = mipLevels;
        imageInfo.arrayLayers   = 1;
        imageInfo.format        = format;
        imageInfo.tiling        = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage         = usage;
        imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples       = numSamples;
        imageInfo.flags         = 0; // Optional

        if (vkCreateImage(logicalDevice, &imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(logicalDevice, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = find_memory_type(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(logicalDevice, image, imageMemory, 0);
    }

    Void transition_image_layout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, UInt32 mipLevels)
	{
        VkCommandBuffer commandBuffer;
    	begin_single_time_commands(commandBuffer);

        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout                       = oldLayout;
        barrier.newLayout                       = newLayout;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = image;
        if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) 
        {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (has_stencil_component(format))
            {
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        } else {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask                   = 0; // TODO
        barrier.dstAccessMask                   = 0; // TODO

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && 
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) 
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
    	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && 
                 newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) 
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
    	else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && 
                 newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) 
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(commandBuffer,
                             sourceStage,
                             destinationStage,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &barrier);

        end_single_time_commands(commandBuffer);
    }

    Void copy_buffer_to_image(VkBuffer buffer, VkImage image, UInt32 width, UInt32 height)
	{
        VkCommandBuffer commandBuffer;
    	begin_single_time_commands(commandBuffer);

        VkBufferImageCopy region{};
        region.bufferOffset                    = 0;
        region.bufferRowLength                 = 0;
        region.bufferImageHeight               = 0;
        region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel       = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount     = 1;
        region.imageOffset                     = { 0, 0, 0 };
        region.imageExtent                     = { width, height,1 };

        vkCmdCopyBufferToImage(commandBuffer,
                               buffer,
                               image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1,
                               &region);

        end_single_time_commands(commandBuffer);
    }

    Void generate_mipmaps(VkImage image, VkFormat imageFormat, Int32 texWidth, Int32 texHeight, UInt32 mipLevels)
	{

        // Check if image format supports linear blitting
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        {
            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        VkCommandBuffer commandBuffer;
    	begin_single_time_commands(commandBuffer);

        VkImageMemoryBarrier barrier{};
        barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image                           = image;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.subresourceRange.levelCount     = 1;

        Int32 mipWidth  = texWidth;
        Int32 mipHeight = texHeight;

        for (UInt32 i = 1; i < mipLevels; ++i)
        {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout                     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout                     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask                 = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask                 = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 
                                 0,
                                 0, 
                                 nullptr,
                                 0, 
                                 nullptr,
                                 1, 
                                 &barrier);

            VkImageBlit blit{};
            blit.srcOffsets[0]                 = { 0, 0, 0 };
            blit.srcOffsets[1]                 = { mipWidth, mipHeight, 1 };
            blit.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel       = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount     = 1;
            blit.dstOffsets[0]                 = { 0, 0, 0 };
            blit.dstOffsets[1]                 = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
            blit.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel       = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount     = 1;

            vkCmdBlitImage(commandBuffer,
                           image, 
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           image, 
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, 
                           &blit,
                           VK_FILTER_LINEAR);

            barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0,
                                 0, 
                                 nullptr,
                                 0, 
                                 nullptr,
                                 1, 
                                 &barrier);

            if (mipWidth > 1)
            {
                mipWidth /= 2;
            }
            if (mipHeight > 1)
            {
                mipHeight /= 2;
            }

        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
                             0,
                             0, 
                             nullptr,
                             0, 
                             nullptr,
                             1, 
                             &barrier);

        end_single_time_commands(commandBuffer);
    }

    Void create_texture_image_view()
	{
        textureImageView = create_image_view(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
    }

    VkImageView create_image_view(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, UInt32 mipLevels)
	{
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image                           = image;
        viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format                          = format;
        viewInfo.subresourceRange.aspectMask     = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        VkImageView imageView;
        if (vkCreateImageView(logicalDevice, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }

    Void create_texture_sampler()
	{
        VkSamplerCreateInfo samplerInfo{};
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
			
        samplerInfo.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter               = VK_FILTER_LINEAR;
        samplerInfo.minFilter               = VK_FILTER_LINEAR;
        samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable        = VK_TRUE;
        samplerInfo.maxAnisotropy           = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable           = VK_FALSE;
        samplerInfo.compareOp               = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias              = 0.0f;
        samplerInfo.minLod                  = 0.0f;
        samplerInfo.maxLod                  = static_cast<Float32>(mipLevels);

        if (vkCreateSampler(logicalDevice, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    Void load_model()
	{
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, (MODELS_PATH + "viking_room/viking_room.obj").c_str()))
        {
            throw std::runtime_error(warn + err);
        }

        std::unordered_map<Vertex, UInt32> uniqueVertices{};
        for (const tinyobj::shape_t &shape : shapes) 
        {
            for (const tinyobj::index_t &index : shape.mesh.indices) 
            {
                Vertex vertex{};
                vertex.position = 
                {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.uv = 
                {
                    attrib.texcoords[2 * index.texcoord_index + 0],
                    attrib.texcoords[2 * index.texcoord_index + 1]
                };

                vertex.color = { 1.0f, 1.0f, 1.0f };

                if (!uniqueVertices.contains(vertex))
                {
                    uniqueVertices[vertex] = static_cast<UInt32>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }
    }

    Void create_vertex_buffer()
	{
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        create_buffer(bufferSize, 
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                     stagingBuffer,
                     stagingBufferMemory);

        Void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (UInt64)bufferSize);
        vkUnmapMemory(logicalDevice, stagingBufferMemory);


        create_buffer(bufferSize, 
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                     vertexBuffer, 
                     vertexBufferMemory);

        copy_buffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    Void create_index_buffer()
	{
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        create_buffer(bufferSize, 
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      stagingBuffer, 
                      stagingBufferMemory);

        Void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        create_buffer(bufferSize, 
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      indexBuffer,
                      indexBufferMemory);

        copy_buffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    Void create_uniform_buffers()
	{
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (UInt64 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) 
        {
            create_buffer(bufferSize, 
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, 
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          uniformBuffers[i], 
                          uniformBuffersMemory[i]);

            vkMapMemory(logicalDevice, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }

    Void create_shader_storage_buffers()
    {
        // Initialize particles
        std::default_random_engine rndEngine(UInt32(time(nullptr)));
        std::uniform_real_distribution<Float32> rndDist(0.0f, 1.0f);

        // Initial particle positions on a circle
        std::vector<Particle> particles(PARTICLE_COUNT);
        for (Particle &particle : particles) 
        {
            Float32 r = 0.25f * sqrt(rndDist(rndEngine));
            Float32 theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
            Float32 x = r * cos(theta) * HEIGHT / WIDTH;
            Float32 y = r * sin(theta);
            particle.position = glm::vec2(x, y);
            particle.velocity = glm::normalize(glm::vec2(x, y)) * 0.00025f;
            particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
        }

        VkDeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        create_buffer(bufferSize, 
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
                      stagingBuffer, 
                      stagingBufferMemory);

        Void *data;
        vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, particles.data(), (UInt64)bufferSize);
        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        shaderStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        shaderStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

        for (UInt64 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) 
        {
            create_buffer(bufferSize,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
                          shaderStorageBuffers[i], 
                          shaderStorageBuffersMemory[i]);
            // Copy data from the staging buffer (host) to the shader storage buffer (GPU)
            copy_buffer(stagingBuffer, shaderStorageBuffers[i], bufferSize);
        }

        vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
        vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
    }

    Void create_descriptor_pool()
	{
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<UInt32>(MAX_FRAMES_IN_FLIGHT);

        //poolSizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        //poolSizes[1].descriptorCount = static_cast<UInt32>(MAX_FRAMES_IN_FLIGHT);

        poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = static_cast<UInt32>(MAX_FRAMES_IN_FLIGHT) * 2U;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<UInt32>(poolSizes.size());
        poolInfo.pPoolSizes    = poolSizes.data();
        poolInfo.maxSets       = static_cast<UInt32>(MAX_FRAMES_IN_FLIGHT);

        if (vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    Void create_descriptor_sets()
	{
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<UInt32>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts        = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

        if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (UInt64 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            VkDescriptorBufferInfo uniformBufferInfo{};
            uniformBufferInfo.buffer = uniformBuffers[i];
            uniformBufferInfo.offset = 0;
            uniformBufferInfo.range  = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView   = textureImageView;
            imageInfo.sampler     = textureSampler;

            VkDescriptorBufferInfo storageBufferInfoLastFrame{};
            storageBufferInfoLastFrame.buffer = shaderStorageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT];
            storageBufferInfoLastFrame.offset = 0;
            storageBufferInfoLastFrame.range  = sizeof(Particle) * PARTICLE_COUNT;

            VkDescriptorBufferInfo storageBufferInfoCurrentFrame{};
            storageBufferInfoCurrentFrame.buffer = shaderStorageBuffers[i];
            storageBufferInfoCurrentFrame.offset = 0;
            storageBufferInfoCurrentFrame.range  = sizeof(Particle) * PARTICLE_COUNT;

            std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
            
            descriptorWrites[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet           = descriptorSets[i];
            descriptorWrites[0].dstBinding       = 0;
            descriptorWrites[0].dstArrayElement  = 0;
            descriptorWrites[0].descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount  = 1;
            descriptorWrites[0].pBufferInfo      = &uniformBufferInfo;

            descriptorWrites[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet           = descriptorSets[i];
            descriptorWrites[1].dstBinding       = 1;
            descriptorWrites[1].dstArrayElement  = 0;
            descriptorWrites[1].descriptorType   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount  = 1;
            descriptorWrites[1].pImageInfo       = &imageInfo;

            descriptorWrites[2].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet           = descriptorSets[i];
            descriptorWrites[2].dstBinding       = 2;
            descriptorWrites[2].dstArrayElement  = 0;
            descriptorWrites[2].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2].descriptorCount  = 1;
            descriptorWrites[2].pBufferInfo      = &storageBufferInfoLastFrame;

            descriptorWrites[3].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet           = descriptorSets[i];
            descriptorWrites[3].dstBinding       = 3;
            descriptorWrites[3].dstArrayElement  = 0;
            descriptorWrites[3].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[3].descriptorCount  = 1;
            descriptorWrites[3].pBufferInfo      = &storageBufferInfoCurrentFrame;

            vkUpdateDescriptorSets(logicalDevice, 
                                   static_cast<UInt32>(descriptorWrites.size()),
                                   descriptorWrites.data(),
                                   0, 
                                   nullptr);
        }
    }

    Void create_buffer(VkDeviceSize size, 
                       VkBufferUsageFlags usage, 
                       VkMemoryPropertyFlags properties, 
                       VkBuffer &buffer, 
                       VkDeviceMemory &bufferMemory)
	{
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size        = size;
        bufferInfo.usage       = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memRequirements.size;
        allocInfo.memoryTypeIndex = find_memory_type(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0);
    }

    Void copy_buffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
        VkCommandBuffer commandBuffer;
        begin_single_time_commands(commandBuffer);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0; // Optional
        copyRegion.dstOffset = 0; // Optional
        copyRegion.size      = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        end_single_time_commands(commandBuffer);
    }

    Void begin_single_time_commands(VkCommandBuffer &commandBuffer)
	{
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool        = commandPool;
        allocInfo.commandBufferCount = 1;
        
        vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
    }

    Void end_single_time_commands(VkCommandBuffer commandBuffer)
	{
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);
    }

    UInt32 find_memory_type(UInt32 typeFilter, VkMemoryPropertyFlags properties)
	{
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) 
        {
            if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    Void create_command_buffers()
	{
        graphicsCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = commandPool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = UInt32(graphicsCommandBuffers.size());

        if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, graphicsCommandBuffers.data()) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    Void create_compute_command_buffers()
	{
        computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = commandPool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = UInt32(computeCommandBuffers.size());

        if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, computeCommandBuffers.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    Void create_synchronization_objects()
	{
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (UInt64 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
        {
            if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS ||
                vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(logicalDevice, &fenceInfo, nullptr, &computeInFlightFences[i]) != VK_SUCCESS)
            {

                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    //TODO: Rate device in the future
    Int32 rate_device_suitability(VkPhysicalDevice device)
	{
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        Int32 score = 0;

        // Discrete GPUs have a significant performance advantage
        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) 
        {
            score += 1000;
        }

        // Maximum possible size of textures affects graphics quality
        score += deviceProperties.limits.maxImageDimension2D;

        // Application can't function without geometry shaders
        if (!deviceFeatures.geometryShader) 
        {
            return 0;
        }

        return score;
    }

    Void update()
	{
        Float32 lastTime = 0.0f;
        while (!glfwWindowShouldClose(window)) 
        {
            glfwPollEvents();
            draw_frame();
            const Float32 currentTime = glfwGetTime();
            lastFrameTime = (currentTime - lastTime) * 1000.0f;
            lastTime = currentTime;
        }

        vkDeviceWaitIdle(logicalDevice);
    }

	Void draw_frame()
	{
        vkWaitForFences(logicalDevice, 1, &computeInFlightFences[currentFrame], VK_TRUE, std::numeric_limits<UInt64>::max());

        update_uniform_buffer(currentFrame);

        vkResetFences(logicalDevice, 1, &computeInFlightFences[currentFrame]);

        vkResetCommandBuffer(computeCommandBuffers[currentFrame], 0);
        record_compute_command_buffer(computeCommandBuffers[currentFrame]);

        VkSubmitInfo submitInfo{};
        submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount   = 1;
        submitInfo.pCommandBuffers      = &computeCommandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores    = &computeFinishedSemaphores[currentFrame];

        if (vkQueueSubmit(computeQueue, 1, &submitInfo, computeInFlightFences[currentFrame]) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to submit compute command buffer!");
        }

        vkWaitForFences(logicalDevice, 1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<UInt64>::max());

        UInt32 imageIndex;
    	VkResult result = vkAcquireNextImageKHR(logicalDevice, 
                                                swapChain, 
                                                std::numeric_limits<UInt64>::max(), 
                                                imageAvailableSemaphores[currentFrame], 
                                                VK_NULL_HANDLE,
                                                &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreate_swap_chain();
            return;
        }
    	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) 
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

    	vkResetFences(logicalDevice, 1, &inFlightFences[currentFrame]);
        vkResetCommandBuffer(graphicsCommandBuffers[currentFrame], 0);
        record_command_buffer(graphicsCommandBuffers[currentFrame], imageIndex);

        std::array<VkSemaphore, 2> waitSemaphores      = { computeFinishedSemaphores[currentFrame], imageAvailableSemaphores[currentFrame] };
        std::array<VkSemaphore, 1> signalSemaphores    = { renderFinishedSemaphores[currentFrame] };
        std::array<VkPipelineStageFlags, 2> waitStages = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

        submitInfo = {};
        submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount   = static_cast<UInt32>(waitSemaphores.size());
        submitInfo.pWaitSemaphores      = waitSemaphores.data();
        submitInfo.pWaitDstStageMask    = waitStages.data();
        submitInfo.commandBufferCount   = 1;
        submitInfo.pCommandBuffers      = &graphicsCommandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = static_cast<UInt32>(signalSemaphores.size());
        submitInfo.pSignalSemaphores    = signalSemaphores.data();

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        std::array<VkSwapchainKHR, 1> swapChains = { swapChain };
        presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = static_cast<UInt32>(signalSemaphores.size());
        presentInfo.pWaitSemaphores    = signalSemaphores.data();
        presentInfo.swapchainCount     = static_cast<UInt32>(swapChains.size());
        presentInfo.pSwapchains        = swapChains.data();
        presentInfo.pImageIndices      = &imageIndex;
        presentInfo.pResults           = nullptr; // Optional
        
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
        {
            framebufferResized = false;
            recreate_swap_chain();
        }
    	else if (result != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

    Void record_command_buffer(VkCommandBuffer commandBuffer, UInt32 imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags            = 0; // Optional
        beginInfo.pInheritanceInfo = nullptr; // Optional

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        std::array<VkClearValue, 3> clearValues{};
        clearValues[0].color        = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[2].depthStencil = { 1.0f, 0 };
        
        renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass        = renderPass;
        renderPassInfo.framebuffer       = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
        renderPassInfo.clearValueCount   = static_cast<UInt32>(clearValues.size());
        renderPassInfo.pClearValues      = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

	        //VkBuffer vertexBuffers[] = { vertexBuffer };
	        //VkDeviceSize offsets[] = { 0 };
	        //vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
	        //vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

	        VkViewport viewport{};
	        viewport.x        = 0.0f;
	        viewport.y        = 0.0f;
	        viewport.width    = static_cast<Float32>(swapChainExtent.width);
	        viewport.height   = static_cast<Float32>(swapChainExtent.height);
	        viewport.minDepth = 0.0f;
	        viewport.maxDepth = 1.0f;
	        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	        VkRect2D scissor{};
	        scissor.offset = { 0, 0 };
	        scissor.extent = swapChainExtent;
	        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);


            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &shaderStorageBuffers[currentFrame], offsets);

            vkCmdDraw(commandBuffer, PARTICLE_COUNT, 1, 0, 0);

	        //vkCmdBindDescriptorSets(commandBuffer, 
	        //                        VK_PIPELINE_BIND_POINT_GRAPHICS, 
	        //                        graphicsPipelineLayout, 
	        //                        0, 
	        //                        1,
	        //                        &descriptorSets[currentFrame],
	        //                        0, 
	        //                        nullptr);
            //
	        //vkCmdDrawIndexed(commandBuffer, 
	        //                 static_cast<UInt32>(indices.size()), 
	        //                 1,
	        //                 0, 
	        //                 0, 
	        //                 0);

    	vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    Void record_compute_command_buffer(VkCommandBuffer commandBuffer)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags            = 0; // Optional
        beginInfo.pInheritanceInfo = nullptr; // Optional

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

        vkCmdBindDescriptorSets(commandBuffer, 
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                computePipelineLayout, 
                                0, 
                                1,
                                &descriptorSets[currentFrame],
                                0, 
                                nullptr);

        vkCmdDispatch(commandBuffer, PARTICLE_COUNT / 256, 1, 1);
        

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    Void recreate_swap_chain()
	{
        Int32 width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) 
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(logicalDevice);

        cleanup_swap_chain();

        create_swap_chain();
        create_image_views();
        create_color_resources();
        create_depth_resources();
        create_framebuffers();
    }

    Void update_uniform_buffer(UInt32 currentImage)
	{
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        Float32 time = std::chrono::duration<Float32, std::chrono::seconds::period>(currentTime - startTime).count();
        UniformBufferObject ubo{};
        //ubo.model = glm::rotate(glm::mat4(1.0f),
        //                        time * glm::radians(30.0f), 
        //                        glm::vec3(0.0f, 0.0f, 1.0f));
        ////ubo.model = glm::scale(ubo.model, glm::vec3(2.0f));
        //ubo.view = glm::lookAt(glm::vec3(2.0f), 
        //                       glm::vec3(0.0f, 0.0f, 0.0f), 
        //                       glm::vec3(0.0f, 0.0f, 1.0f));
        //
        //ubo.projection = glm::perspective(glm::radians(45.0f),
        //                                  swapChainExtent.width / Float32(swapChainExtent.height), 
        //                                  0.1f,
        //                                  10.0f);
        //ubo.projection[1][1] *= -1.0f;
        ubo.deltaTime = lastFrameTime * 2.0f;
        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL s_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                           VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                           const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                                                           Void *pUserData)
    {
        switch (messageSeverity)
        {
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            {
                SPDLOG_INFO("Validation layer: {}", pCallbackData->pMessage);
                break;
            }
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            {
                SPDLOG_WARN("Validation layer: {}", pCallbackData->pMessage);
                break;
            }
            case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            {
                SPDLOG_ERROR("Validation layer: {}", pCallbackData->pMessage);
                break;
            }
            default:
            {
                SPDLOG_INFO("Not supported severity!");
                break;
            }
        }

        return VK_FALSE;
    }

    static Void s_framebuffer_resize_callback(GLFWwindow *window, Int32 width, Int32 height)
	{
        HelloTriangleApplication* app = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    Void destroy_debug_utils_messenger_ext(VkInstance instance,
                                           VkDebugUtilsMessengerEXT debugMessenger,
                                           const VkAllocationCallbacks *pAllocator)
    {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr)
        {
            func(instance, debugMessenger, pAllocator);
        }
    }

    Void cleanup_swap_chain()
    {
        vkDestroyImageView(logicalDevice, colorImageView, nullptr);
        vkDestroyImage(logicalDevice, colorImage, nullptr);
        vkFreeMemory(logicalDevice, colorImageMemory, nullptr);
        
        vkDestroyImageView(logicalDevice, depthImageView, nullptr);
        vkDestroyImage(logicalDevice, depthImage, nullptr);
        vkFreeMemory(logicalDevice, depthImageMemory, nullptr);

        for (VkFramebuffer &framebuffer : swapChainFramebuffers)
        {
            vkDestroyFramebuffer(logicalDevice, framebuffer, nullptr);
        }

        for (VkImageView &imageView : swapChainImageViews)
        {
            vkDestroyImageView(logicalDevice, imageView, nullptr);
        }

        vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);
    }

    Void shutdown()
	{
        cleanup_swap_chain();

        vkDestroySampler(logicalDevice, textureSampler, nullptr);
        vkDestroyImageView(logicalDevice, textureImageView, nullptr);
        vkDestroyImage(logicalDevice, textureImage, nullptr);
        vkFreeMemory(logicalDevice, textureImageMemory, nullptr);

        for (UInt64 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) 
        {
            vkDestroyBuffer(logicalDevice, shaderStorageBuffers[i], nullptr);
            vkFreeMemory(logicalDevice, shaderStorageBuffersMemory[i], nullptr);
        }

        for (UInt64 i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) 
        {
            vkDestroyBuffer(logicalDevice, uniformBuffers[i], nullptr);
            vkFreeMemory(logicalDevice, uniformBuffersMemory[i], nullptr);
        }
        vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);

        vkDestroyBuffer(logicalDevice, indexBuffer, nullptr);
        vkFreeMemory(logicalDevice, indexBufferMemory, nullptr);

        vkDestroyBuffer(logicalDevice, vertexBuffer, nullptr);
        vkFreeMemory(logicalDevice, vertexBufferMemory, nullptr);

        vkDestroyPipeline(logicalDevice, computePipeline, nullptr);
        vkDestroyPipelineLayout(logicalDevice, computePipelineLayout, nullptr);

        vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(logicalDevice, graphicsPipelineLayout, nullptr);
        vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

        for (UInt64 i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
        {
            vkDestroySemaphore(logicalDevice, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(logicalDevice, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(logicalDevice, inFlightFences[i], nullptr);
            vkDestroySemaphore(logicalDevice, computeFinishedSemaphores[i], nullptr);
            vkDestroyFence(logicalDevice, computeInFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(logicalDevice, commandPool, nullptr);
        vkDestroyDevice(logicalDevice, nullptr);

        if (ENABLE_VALIDATION_LAYERS) 
        {
            destroy_debug_utils_messenger_ext(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

Int32 main()
{
    HelloTriangleApplication app;

    try 
    {
        app.run();
    }
	catch (const std::exception &e) 
    {
        SPDLOG_ERROR("{}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}