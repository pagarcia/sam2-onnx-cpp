#ifdef __APPLE__
#import <Cocoa/Cocoa.h>
#include "openFileDialog.h"
#include <string>

std::string openFileDialog() {
    @autoreleasepool {
        NSOpenPanel* panel = [NSOpenPanel openPanel];
        [panel setCanChooseFiles:YES];
        [panel setCanChooseDirectories:NO];
        [panel setAllowsMultipleSelection:NO];
        [panel setAllowedFileTypes:@[@"jpg", @"jpeg", @"png", @"bmp"]];
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
