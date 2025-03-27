// openFileDialog.h

#ifndef OPENFILEDIALOG_H
#define OPENFILEDIALOG_H

#include <string>

/**
 * Opens a file dialog with a specified filter (for file extensions)
 * and a dialog box title. Returns the selected file path (UTF-8),
 * or an empty string if canceled.
 *
 * On Windows, the 'filter' parameter must be a double-null-terminated wide string,
 * e.g. L"Image Files\0*.jpg;*.jpeg;*.png;*.bmp\0All Files\0*.*\0"
 *
 * On macOS, you can parse the string inside openFileDialog.mm if needed.
 */
std::string openFileDialog(const wchar_t* filter = L"All Files\0*.*\0",
                           const wchar_t* title  = L"Select a File");

#endif // OPENFILEDIALOG_H
