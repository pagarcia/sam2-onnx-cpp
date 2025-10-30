# sam2-onnx-cpp/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(SAM2_ONNX)

set(TARGET_NAME "Segment")

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output exe into build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# QoL: compiler flags
if(MSVC)
  add_definitions(-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
  add_compile_options(/O2)
else()
  add_compile_options(-O3 -fno-omit-frame-pointer)
endif()

# ---- Inputs users must provide ----
if(NOT DEFINED OpenCV_DIR)
  message(FATAL_ERROR "Please set OpenCV_DIR (e.g., -DOpenCV_DIR=\"/opt/homebrew/opt/opencv\")")
endif()

if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Please set ONNXRUNTIME_DIR (e.g., -DONNXRUNTIME_DIR=\"/opt/onnxruntime-osx-arm64-1.23.2\")")
endif()

# ---- 1) OpenCV ----
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV ${OpenCV_VERSION} at: ${OpenCV_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})

# ---- 2) ONNX Runtime headers + library ----
include_directories(${ONNXRUNTIME_DIR}/include)

# Find the ORT import library/shared object in a version-agnostic way
if(WIN32)
  # onnxruntime.lib (import lib); DLLs will be handled below
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED)
elseif(APPLE)
  # libonnxruntime.dylib (may be a symlink to libonnxruntime.<ver>.dylib)
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED)

  # Ensure bundle+rpaths so dyld looks in .app/Contents/Frameworks
  set(CMAKE_MACOSX_BUNDLE YES)
  set(CMAKE_BUILD_RPATH   "@loader_path/../Frameworks")
  set(CMAKE_INSTALL_RPATH "@loader_path/../Frameworks;@executable_path/../Frameworks")
else()
  # Linux
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED)
  # Let the loader look next to the binary as well
  set(CMAKE_BUILD_RPATH   "$ORIGIN")
  set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()

message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")

# ---- 3) Sources ----
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR})  # headers at repo root

if(WIN32)
  set(OPENFILEDIALOG_SRC src/openFileDialog.cpp)
elseif(APPLE)
  set(OPENFILEDIALOG_SRC src/openFileDialog.mm)
  # Enable ARC for ObjC++
  set_source_files_properties(src/openFileDialog.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
  set(OPENFILEDIALOG_SRC "")
endif()

add_executable(${TARGET_NAME}
  src/main.cpp
  src/onnx_test_image.cpp
  src/onnx_test_video.cpp
  src/SAM2Session.cpp
  src/SAM2Image.cpp
  src/SAM2Video.cpp
  ${OPENFILEDIALOG_SRC}
)

# ---- 4) Link ----
if(APPLE)
  target_link_libraries(${TARGET_NAME} PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
    "-framework Cocoa"
  )

  # Make sure the executable will search ../Frameworks at runtime
  set_target_properties(${TARGET_NAME} PROPERTIES
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/Info.plist.in"
    BUILD_RPATH   "@loader_path/../Frameworks"
    INSTALL_RPATH "@loader_path/../Frameworks;@executable_path/../Frameworks;${CMAKE_INSTALL_RPATH}"
  )

elseif(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
  )

  # Convenience: copy all ONNX Runtime runtime DLLs beside the exe after build
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying ONNX Runtime DLLs..."
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${ONNXRUNTIME_DIR}/lib"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>"
    # Filter to only DLLs named onnxruntime*.dll
    COMMAND ${CMAKE_COMMAND} -E echo "Filtering copied files to onnxruntime*.dll is recommended when packaging."
  )

else()
  target_link_libraries(${TARGET_NAME} PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
  )
endif()

# ---- 5) Install / package ----
if(APPLE)
  # Install the app bundle
  install(TARGETS ${TARGET_NAME}
          BUNDLE DESTINATION .
          RUNTIME DESTINATION bin)

  # Put models next to the binary inside the .app (Contents/MacOS)
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/models/"
          DESTINATION "${TARGET_NAME}.app/Contents/MacOS")

  # 🚩 Critical: install *all* ORT dylibs (versioned + unversioned) into Frameworks
  file(GLOB ORT_DYLIBS "${ONNXRUNTIME_DIR}/lib/libonnxruntime*.dylib")
  if(ORT_DYLIBS)
    install(FILES ${ORT_DYLIBS}
            DESTINATION "${TARGET_NAME}.app/Contents/Frameworks")
  else()
    message(WARNING "No libonnxruntime*.dylib found under ${ONNXRUNTIME_DIR}/lib")
  endif()

  # DMG packaging
  include(InstallRequiredSystemLibraries)
  set(CPACK_PACKAGE_NAME "SAM2_ONNX")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_GENERATOR "DragNDrop")
  include(CPack)

elseif(WIN32)
  install(TARGETS ${TARGET_NAME}
          RUNTIME DESTINATION bin)

  # Ship models for convenience
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/models/"
          DESTINATION share/models)

  # Optionally install ORT runtime DLLs
  install(DIRECTORY "${ONNXRUNTIME_DIR}/lib/"
          DESTINATION bin
          FILES_MATCHING PATTERN "onnxruntime*.dll")

  set(CPACK_PACKAGE_NAME "SAM2_ONNX")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_GENERATOR "ZIP")
  include(CPack)

else() # Linux / others
  install(TARGETS ${TARGET_NAME}
          RUNTIME DESTINATION bin)

  install(DIRECTORY "${CMAKE_SOURCE_DIR}/models/"
          DESTINATION share/models)

  set(CPACK_PACKAGE_NAME "SAM2_ONNX")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_GENERATOR "TGZ")
  include(CPack)
endif()
