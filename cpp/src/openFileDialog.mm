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

    // Convert wide -> std::string (UTF-8) for easier splitting
    // But we can do splitting in wide if we prefer. 
    // Let's do wide splitting.
    std::wstring delim = L";";
    size_t start = 0;
    while (true) {
        size_t pos = patternLine.find(delim, start);
        std::wstring token = (pos == std::wstring::npos)
            ? patternLine.substr(start)
            : patternLine.substr(start, pos - start);

        // trim any whitespace (optional)
        // remove leading "*." if present
        if (token.size() > 2 && token[0] == L'*' && token[1] == L'.') {
            token = token.substr(2); // skip "*."
        }
        // else if token starts with ".", remove that => e.g. ".jpg"
        else if (!token.empty() && token[0] == L'.') {
            token = token.substr(1);
        }

        // convert to UTF-8
        if (!token.empty()) {
            // Note: might contain wildcard leftover if user typed something else
            // but typically it's just e.g. "jpg" or "png"
            int needed = (int)wcstombs(nullptr, token.c_str(), 0);
            if (needed > 0) {
                std::string ext(needed, '\0');
                wcstombs(&ext[0], token.c_str(), needed+1);
                // lower-case it? e.g. ext = toLowerCase(ext);
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
 * to Windows, but on mac we build an NSArray of file extensions for [panel setAllowedFileTypes:].
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

        // Convert 'title' to NSString
        if (title) {
            NSString* nsTitle = [NSString stringWithCharacters:title length:wcslen(title)];
            [panel setTitle:nsTitle];
        } else {
            [panel setTitle:@"Open File"];
        }

        // Parse the double-null-terminated filter
        std::vector<std::wstring> segments = splitDoubleNullTerminated(filter);
        // Typically we have pairs of (description, patternLine).
        // We'll gather all extensions in a single list for allowedFileTypes.
        // If we find "*.*", we might interpret that as "All Files" => don't restrict.

        NSMutableSet<NSString*>* extensionSet = [NSMutableSet set];

        for (size_t i = 0; i+1 < segments.size(); i += 2) {
            // segments[i]   => description (e.g. "Image Files")
            // segments[i+1] => patternLine (e.g. "*.jpg;*.png;*.bmp")
            std::wstring patternLine = segments[i+1];
            // check if it's "*.*" => that means "All Files"
            if (patternLine.find(L"*.*") != std::wstring::npos) {
                // Means "All Files" => we can skip setting allowedFileTypes,
                // or we handle it specially. For now, skip => user can open anything.
                extensionSet = nil; // nil => no restriction
                break;
            }

            // otherwise parse
            std::vector<std::string> exts = parseExtensions(patternLine);
            for (auto &e : exts) {
                // e might be "jpg" or "png" etc.
                NSString* nsExt = [NSString stringWithUTF8String:e.c_str()];
                [extensionSet addObject:nsExt];
            }
        }

        if (extensionSet != nil && extensionSet.count > 0) {
            [panel setAllowedFileTypes:[extensionSet allObjects]];
        } else {
            // means "All Files" or empty => no restriction
            // [panel setAllowedFileTypes:nil];
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
