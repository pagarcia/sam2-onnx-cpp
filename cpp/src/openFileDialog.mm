// sam2-onnx-cpp/cpp/src/openFileDialog.mm
#ifdef __APPLE__
#import <Cocoa/Cocoa.h>
#import <AppKit/AppKit.h>
#import <dispatch/dispatch.h>
#include "openFileDialog.h"
#include <string>
#include <vector>

// Split a double-null-terminated wide string into segments.
static std::vector<std::wstring> splitDoubleNullTerminated(const wchar_t* wstr) {
    std::vector<std::wstring> result;
    if (!wstr) return result;
    const wchar_t* p = wstr;
    while (*p != L'\0') {
        const wchar_t* start = p;
        while (*p != L'\0') ++p;
        result.emplace_back(start, p);
        ++p; // skip the '\0'
    }
    return result;
}

// Parse something like "*.jpg;*.jpeg;*.png" -> ["jpg","jpeg","png"] (utf8)
static std::vector<std::string> parseExtensions(const std::wstring &patternLine) {
    std::vector<std::string> exts;
    size_t start = 0;
    while (true) {
        size_t pos = patternLine.find(L';', start);
        std::wstring token = (pos == std::wstring::npos) ? patternLine.substr(start)
                                                         : patternLine.substr(start, pos - start);
        // Trim "*." or "." prefix
        if (token.size() > 2 && token[0] == L'*' && token[1] == L'.') token = token.substr(2);
        else if (!token.empty() && token[0] == L'.') token = token.substr(1);

        if (!token.empty()) {
            size_t need = wcstombs(nullptr, token.c_str(), 0);
            if (need != (size_t)-1 && need > 0) {
                std::string utf8(need, '\0');
                wcstombs(&utf8[0], token.c_str(), need + 1);
                exts.push_back(utf8);
            }
        }
        if (pos == std::wstring::npos) break;
        start = pos + 1;
    }
    return exts;
}

std::string openFileDialog(const wchar_t* filter, const wchar_t* title)
{
    __block std::string resultPath;

    auto presentBlock = ^{
        // Ensure NSApplication exists and we’re foreground
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp activateIgnoringOtherApps:YES];

        NSOpenPanel* panel = [NSOpenPanel openPanel];
        [panel setCanChooseFiles:YES];
        [panel setCanChooseDirectories:NO];
        [panel setAllowsMultipleSelection:NO];
        [panel setResolvesAliases:YES];

        // Title
        if (title && wcslen(title) > 0) {
            size_t need = wcstombs(nullptr, title, 0);
            if (need != (size_t)-1 && need > 0) {
                std::string utf8(need, '\0');
                wcstombs(&utf8[0], title, need + 1);
                [panel setTitle:[NSString stringWithUTF8String:utf8.c_str()]];
            }
        } else {
            [panel setTitle:@"Select a File"];
        }

        // Parse filter → allowed extensions (optional)
        NSMutableSet<NSString*>* extsSet = [NSMutableSet set];
        if (filter) {
            auto segs = splitDoubleNullTerminated(filter);
            for (size_t i = 0; i + 1 < segs.size(); i += 2) {
                const auto& pattern = segs[i + 1];
                if (pattern.find(L"*.*") != std::wstring::npos) { extsSet = nil; break; }
                for (auto &e : parseExtensions(pattern)) {
                    [extsSet addObject:[NSString stringWithUTF8String:e.c_str()]];
                }
            }
        }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        if (extsSet != nil && extsSet.count > 0)
            [panel setAllowedFileTypes:[extsSet allObjects]];
#pragma clang diagnostic pop

        NSInteger res = [panel runModal];
        if (res == NSModalResponseOK) {
            NSURL* url = [[panel URLs] firstObject];
            if (url) resultPath = std::string([[url path] UTF8String]);
            else     resultPath.clear();
        } else {
            resultPath.clear();
        }
    };

    // Must be on main thread for AppKit UI
    if ([NSThread isMainThread]) {
        presentBlock();
    } else {
        dispatch_sync(dispatch_get_main_queue(), presentBlock);
    }

    return resultPath;
}
#endif
