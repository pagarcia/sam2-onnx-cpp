#ifdef _WIN32
#include "openFileDialog.h"
#include <windows.h>
#include <commdlg.h>
#include <string>

std::string openFileDialog() {
    OPENFILENAMEW ofn;
    wchar_t fileName[MAX_PATH] = L"";
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = L"Image Files\0*.jpg;*.jpeg;*.png;*.bmp\0All Files\0*.*\0";
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    // Prevent changing the current directory
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrTitle = L"Select an Image";
    
    if (GetOpenFileNameW(&ofn)) {
        int size = WideCharToMultiByte(CP_UTF8, 0, fileName, -1, NULL, 0, NULL, NULL);
        std::string result(size, 0);
        WideCharToMultiByte(CP_UTF8, 0, fileName, -1, &result[0], size, NULL, NULL);
        return result;
    } else {
        return std::string("");
    }
}
#endif
