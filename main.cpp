#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <iostream>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <cstdlib>

#include "Types.hpp"

constexpr UInt32 WIDTH = 800;
constexpr UInt32 HEIGHT = 600;

class HelloTriangleApplication
{
public:
    Void run()
	{
        initWindow();
        initVulkan();
        mainLoop();
        shutdown();
    }

private:
    GLFWwindow *window;
	VkInstance instance;

    Void initWindow()
	{
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    Void initVulkan()
	{
        createInstance();
    }

    Void createInstance()
	{
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Tutorial";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0U, 1U, 0U, 0U);
        appInfo.pEngineName = "RayEngine";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0U, 1U, 0U, 0U);
        appInfo.apiVersion = VK_API_VERSION_1_0;
    }

    Void mainLoop()
	{
        while (!glfwWindowShouldClose(window)) 
        {
            glfwPollEvents();
        }
    }

    Void shutdown()
	{
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