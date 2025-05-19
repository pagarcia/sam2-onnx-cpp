cmake_minimum_required(VERSION 3.14)
project(SAM2_ONNX)

# Define the target name variable (our executable will be "Segment").
set(TARGET_NAME "Segment")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Place the executable in a 'bin' folder within the build directory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Require user to pass OpenCV_DIR.
if(NOT DEFINED OpenCV_DIR)
  message(FATAL_ERROR "Please set OpenCV_DIR when configuring (e.g., -DOpenCV_DIR=\"path/to/opencv\")")
endif()

# Require user to pass ONNXRUNTIME_DIR.
if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Please set ONNXRUNTIME_DIR when configuring (e.g., -DONNXRUNTIME_DIR=\"path/to/onnxruntime\")")
endif()

#
# 1) Find OpenCV
#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#
# 2) Include ONNX Runtime headers + libs
#
if(WIN32)
  include_directories(${ONNXRUNTIME_DIR}/include)
  link_directories(${ONNXRUNTIME_DIR}/lib)
elseif(APPLE)
  include_directories(${ONNXRUNTIME_DIR}/include)
  # For non-standard library locations, you may also need:
  # link_directories(${ONNXRUNTIME_DIR}/lib)
else() # e.g. Linux
  include_directories(${ONNXRUNTIME_DIR}/include)
  # link_directories(${ONNXRUNTIME_DIR}/lib)
endif()

#
# 3) Source files
#
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR})  # for SAM2.h or openFileDialog.h

# Conditionally add the openFileDialog source file (depending on platform).
if(WIN32)
  set(OPENFILEDIALOG_SRC src/openFileDialog.cpp)
elseif(APPLE)
  set(OPENFILEDIALOG_SRC src/openFileDialog.mm)
else()
  # If you have no GUI file dialog on Linux, you could omit or adapt it:
  set(OPENFILEDIALOG_SRC "")
endif()

# Build only main.cpp, onnx_test_image.cpp, SAM2.cpp (plus openFileDialog).
add_executable(${TARGET_NAME}
    src/main.cpp
    src/onnx_test_image.cpp
    src/onnx_test_image_bounding_box.cpp
    src/onnx_test_video.cpp
    src/SAM2Session.cpp
    src/SAM2Image.cpp
    src/SAM2Video.cpp
    ${OPENFILEDIALOG_SRC}
)

#
# 4) Platform-specific linking
#
if(WIN32)
  target_link_libraries(${TARGET_NAME}
      ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
      ${OpenCV_LIBS}
  )
elseif(APPLE)
  # Make the target an .app bundle and set the rpath. This helps it find frameworks/dylibs.
  set_target_properties(${TARGET_NAME} PROPERTIES 
    MACOSX_BUNDLE TRUE
    INSTALL_RPATH "@executable_path/../Frameworks;/opt/homebrew/opt/opencv/lib"
  )

  # Link to the ONNX Runtime dylib, OpenCV libs, and Cocoa (for file dialogs).
  target_link_libraries(${TARGET_NAME}
      ${ONNXRUNTIME_DIR}/lib/libonnxruntime.1.21.0.dylib
      ${OpenCV_LIBS}
      "-framework Cocoa"
  )
else()
  # Linux or others
  target_link_libraries(${TARGET_NAME}
      ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
      ${OpenCV_LIBS}
  )
endif()

#
# 5) Installation & Packaging
#
# ----------------------------------------------------------------------------
#  Mac-Specific Installation Changes
# ----------------------------------------------------------------------------

if(APPLE)
  install(TARGETS ${TARGET_NAME}
          BUNDLE DESTINATION .
          RUNTIME DESTINATION bin
  )
  
  # Instead of installing models/ into Contents/Resources, install them into
  # Contents/MacOS. That way "image_encoder.onnx" is side-by-side with
  # the 'Segment' (or '2D_Segmenter') binary.
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/models/"
          DESTINATION "${TARGET_NAME}.app/Contents/MacOS"
  )

  # Copy the ONNX Runtime .dylib into Contents/Frameworks
  install(FILES "${ONNXRUNTIME_DIR}/lib/libonnxruntime.1.21.0.dylib"
          DESTINATION "${TARGET_NAME}.app/Contents/Frameworks"
  )

  # CPack config for creating a DMG, etc.
  include(InstallRequiredSystemLibraries)
  set(CPACK_PACKAGE_NAME "SAM2_ONNX")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_GENERATOR "DragNDrop")  # Creates a DMG
  include(CPack)

else()
  install(
    TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION bin
  )

  install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/models/"
    DESTINATION share/models
  )

  set(CPACK_PACKAGE_NAME "SAM2_ONNX")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_GENERATOR "TGZ")
  include(CPack)
endif()
