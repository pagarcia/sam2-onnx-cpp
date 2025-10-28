// sam2-onnx-cpp/cpp/src/openFileDialog.cpp
#ifdef _WIN32
#include "openFileDialog.h"
#include <windows.h>
#include <commdlg.h>
#include <string>

std::string openFileDialog(const wchar_t* filter,
                           const wchar_t* title)
{
    OPENFILENAMEW ofn;
    wchar_t fileName[MAX_PATH] = L"";
    ZeroMemory(&ofn, sizeof(ofn));

    ofn.lStructSize  = sizeof(ofn);
    ofn.hwndOwner    = nullptr;
    ofn.lpstrFilter  = filter;       // <== the user-specified filter
    ofn.lpstrFile    = fileName;
    ofn.nMaxFile     = MAX_PATH;
    ofn.Flags        = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    ofn.lpstrTitle   = title;        // <== the user-specified title

    if (GetOpenFileNameW(&ofn)) {
        // Convert wide string to UTF-8
        int size = WideCharToMultiByte(CP_UTF8, 0, fileName, -1, NULL, 0, NULL, NULL);
        std::string result(size, 0);
        WideCharToMultiByte(CP_UTF8, 0, fileName, -1, &result[0], size, NULL, NULL);
        return result;
    } else {
        return std::string("");
    }
}
#endif
