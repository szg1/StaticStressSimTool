#include "utils.h"
#include <cstdio>
#include <string>

// Opens a native macOS file dialog using AppleScript
std::string openFileDialog() {
#ifdef __APPLE__
    std::string result = "";
    const char* cmd = "osascript -e 'try' -e 'POSIX path of (choose file with prompt \"Select an STL file\" of type {\"stl\", \"STL\"})' -e 'end try'";

    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "";

    char buffer[1024];
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result = buffer;
    }
    pclose(pipe);

    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }

    return result;
#else
    printf("File dialog not implemented for non-macOS systems. returning screw.stl\n");
    return "screw.stl"; // Fallback for testing
#endif
}
