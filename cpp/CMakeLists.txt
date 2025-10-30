# sam2-onnx-cpp/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(SAM2_ONNX)

set(TARGET_NAME "Segment")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output exe into build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# QoL: a few sensible flags
if(MSVC)
  add_definitions(-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
  add_compile_options(/O2)
else()
  add_compile_options(-O3 -fno-omit-frame-pointer)
endif()

# Inputs we expect the user to pass
if(NOT DEFINED OpenCV_DIR)
  message(FATAL_ERROR "Please set OpenCV_DIR (e.g., -DOpenCV_DIR=\"path/to/opencv\")")
endif()

if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Please set ONNXRUNTIME_DIR (e.g., -DONNXRUNTIME_DIR=\"path/to/onnxruntime\")")
endif()

# 1) OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 2) ONNX Runtime headers + library (version-agnostic)
include_directories(${ONNXRUNTIME_DIR}/include)

if(WIN32)
  # Finds: C:\...\onnxruntime-win-x64-gpu-1.22.x\lib\onnxruntime.lib
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED)
elseif(APPLE)
  # Finds: /opt/onnxruntime-osx-arm64-1.22.0/lib/libonnxruntime.dylib (or .1.22.0.dylib)
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED)

  # Make app bundle + rpaths so the dylib is found inside the .app
  set(CMAKE_MACOSX_BUNDLE YES)
  set(CMAKE_INSTALL_RPATH "@loader_path/../Frameworks;@executable_path/../Frameworks")
  set(CMAKE_BUILD_RPATH   "@loader_path/../Frameworks")
else()
  # Linux or others
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED)
endif()

# 3) Sources
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR})  # for headers in repo root

if(WIN32)
  set(OPENFILEDIALOG_SRC src/openFileDialog.cpp)
elseif(APPLE)
  set(OPENFILEDIALOG_SRC src/openFileDialog.mm)
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

# 4) Link
if(APPLE)
  target_link_libraries(${TARGET_NAME} PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
    "-framework Cocoa"
  )
elseif(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
  )

  # Optional convenience: copy ORT DLL beside the exe after build so it runs immediately.
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB}"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>"
  )
else()
  target_link_libraries(${TARGET_NAME} PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
  )
endif()

# 5) Install / package
if(APPLE)
  install(TARGETS ${TARGET_NAME}
          BUNDLE DESTINATION .
          RUNTIME DESTINATION bin)

  # Put models next to the binary inside the .app
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/models/"
          DESTINATION "${TARGET_NAME}.app/Contents/MacOS")

  # Install the exact ORT dylib we linked (no hardcoded version)
  get_filename_component(_ORT_NAME "${ONNXRUNTIME_LIB}" NAME)
  install(FILES "${ONNXRUNTIME_DIR}/lib/${_ORT_NAME}"
          DESTINATION "${TARGET_NAME}.app/Contents/Frameworks")

  include(InstallRequiredSystemLibraries)
  set(CPACK_PACKAGE_NAME "SAM2_ONNX")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_GENERATOR "DragNDrop")
  include(CPack)

else()
  install(TARGETS ${TARGET_NAME}
          RUNTIME DESTINATION bin)

  install(DIRECTORY "${CMAKE_SOURCE_DIR}/models/"
          DESTINATION share/models)

  set(CPACK_PACKAGE_NAME "SAM2_ONNX")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_GENERATOR "TGZ")
  include(CPack)
endif()
