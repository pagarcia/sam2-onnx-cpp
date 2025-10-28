// sam2-onnx-cpp/cpp/src/openFileDialog.mm
#ifdef __APPLE__
#import <Cocoa/Cocoa.h>
#include "openFileDialog.h"
#include <string>
#include <vector>

/**
 * A helper function that splits a wide C-string with double-null termination
 * into a list of "segments," each separated by a single '\0'.
 *
 * Example filter:
 *   L"Image Files\0*.jpg;*.jpeg;*.png;*.bmp\0All Files\0*.*\0"
 * yields segments:
 *   [L"Image Files", L"*.jpg;*.jpeg;*.png;*.bmp", L"All Files", L"*.*"]
 */
static std::vector<std::wstring> splitDoubleNullTerminated(const wchar_t* wstr)
{
    std::vector<std::wstring> result;
    if (!wstr) return result;

    const wchar_t* segmentStart = wstr;
    while (*segmentStart != L'\0')
    {
        // find the next '\0'
        const wchar_t* segmentEnd = segmentStart;
        while (*segmentEnd != L'\0') {
            segmentEnd++;
        }
        // now [segmentStart..segmentEnd-1] is one segment
        result.emplace_back(segmentStart, segmentEnd);

        // move to next
        segmentStart = segmentEnd + 1; // skip the '\0'
    }
    return result;
}

/**
 * A helper that parses a pattern line like "*.jpg;*.jpeg;*.png"
 * into extensions: ["jpg","jpeg","png"].
 */
static std::vector<std::string> parseExtensions(const std::wstring &patternLine)
{
    // patternLine might be: L"*.jpg;*.jpeg;*.png;*.bmp"
    // We'll split by ';', remove "*." prefix, keep the extension part
    std::vector<std::string> exts;

    // We'll do wide splitting on ';'
    std::wstring delim = L";";
    size_t start = 0;
    while (true) {
        size_t pos = patternLine.find(delim, start);
        std::wstring token = (pos == std::wstring::npos)
            ? patternLine.substr(start)
            : patternLine.substr(start, pos - start);

        // remove leading "*." if present
        if (token.size() > 2 && token[0] == L'*' && token[1] == L'.') {
            token = token.substr(2); // skip "*."
        }
        // else if token starts with ".", remove that => e.g. ".jpg"
        else if (!token.empty() && token[0] == L'.') {
            token = token.substr(1);
        }

        // convert wide -> UTF-8
        if (!token.empty()) {
            int needed = (int)wcstombs(nullptr, token.c_str(), 0);
            if (needed > 0) {
                std::string ext(needed, '\0');
                wcstombs(&ext[0], token.c_str(), needed+1);
                exts.push_back(ext);
            }
        }

        if (pos == std::wstring::npos) {
            break;
        }
        start = pos + delim.size();
    }
    return exts;
}

/**
 * openFileDialog(...) on macOS with custom filter and title.
 *
 * Filter is a double-null-terminated wide string. We'll parse it similarly
 * to Windows, but on mac we build an NSArray of file extensions for 
 * [panel setAllowedFileTypes:]. This method is deprecated on macOS 12+,
 * but still functional. 
 */
std::string openFileDialog(const wchar_t* filter,
                           const wchar_t* title)
{
    @autoreleasepool {
        NSOpenPanel* panel = [NSOpenPanel openPanel];
        [panel setCanChooseFiles:YES];
        [panel setCanChooseDirectories:NO];
        [panel setAllowsMultipleSelection:NO];
        [panel setResolvesAliases:YES];

        // Convert 'title' (wchar_t*) => UTF-8 => NSString
        if (title) {
            size_t wideLen = wcslen(title);
            if (wideLen == 0) {
                [panel setTitle:@"Open File"];
            } else {
                size_t utf8Size = wcstombs(nullptr, title, 0);
                if (utf8Size == (size_t)-1) {
                    // fallback
                    [panel setTitle:@"Open File"];
                } else {
                    std::vector<char> buf(utf8Size + 1, 0);
                    wcstombs(buf.data(), title, utf8Size + 1);
                    NSString* nsTitle = [NSString stringWithUTF8String:buf.data()];
                    [panel setTitle:nsTitle];
                }
            }
        } else {
            [panel setTitle:@"Open File"];
        }

        // Parse the double-null-terminated filter
        std::vector<std::wstring> segments = splitDoubleNullTerminated(filter);
        // Typically we have pairs of (description, patternLine).
        // We'll gather all extensions in a single list for allowedFileTypes.
        // If we find "*.*", interpret as "All Files" => no restriction.

        NSMutableSet<NSString*>* extensionSet = [NSMutableSet set];

        for (size_t i = 0; i+1 < segments.size(); i += 2) {
            // segments[i]   => description (e.g. "Image Files")
            // segments[i+1] => patternLine (e.g. "*.jpg;*.png;*.bmp")
            std::wstring patternLine = segments[i+1];
            if (patternLine.find(L"*.*") != std::wstring::npos) {
                // Means "All Files" => skip any restriction
                extensionSet = nil;
                break;
            }

            // otherwise parse
            std::vector<std::string> exts = parseExtensions(patternLine);
            for (auto &e : exts) {
                NSString* nsExt = [NSString stringWithUTF8String:e.c_str()];
                [extensionSet addObject:nsExt];
            }
        }

        // setAllowedFileTypes is deprecated in macOS 12+,
        // but works for older versions. We'll silence the warning:
        if (extensionSet != nil && extensionSet.count > 0) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            [panel setAllowedFileTypes:[extensionSet allObjects]];
#pragma clang diagnostic pop
        }

        NSInteger result = [panel runModal];
        if (result == NSModalResponseOK) {
            NSURL* url = [[panel URLs] objectAtIndex:0];
            return std::string([[url path] UTF8String]);
        } else {
            return "";
        }
    }
}
#endif
