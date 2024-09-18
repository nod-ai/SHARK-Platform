# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

# libshortfin publishes multiple python packages: - _shortfin: Trampoline
# __init__.py which looks at environment variables to load an appropriate native
# library. - _shortfin_default.lib: Native library as a default, uninstrumented
# build. - _shortfin_tracing.lib: Native library with tracing enabled. -
# Others.

# nanobind
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        9641bb7151f04120013b812789b3ebdfa7e7324f # 2.1.0
)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(shortfin_python_extension NB_STATIC LTO
  array_binding.cc
  lib_ext.cc
)

if (SHORTFIN_ENABLE_TRACING)
  set_target_properties(shortfin_python_extension
                      PROPERTIES OUTPUT_NAME "_shortfin_tracy/lib")
else()
  set_target_properties(shortfin_python_extension
                      PROPERTIES OUTPUT_NAME "_shortfin_default/lib")
endif()

target_link_libraries(shortfin_python_extension
  PRIVATE ${SHORTFIN_LINK_LIBRARY_NAME}
)

function(shortfin_python_stubs build_type)
  nanobind_add_stub(
    shortfin_python_extension_stub
    MODULE _shortfin_${build_type}.lib
    OUTPUT _shortfin_${build_type}/lib.pyi
    DEPENDS shortfin_python_extension
  )

endfunction()

function(shortfin_python_stubs build_variant)
  set(output_root "${CMAKE_CURRENT_BINARY_DIR}/_shortfin_${build_variant}")
  file(MAKE_DIRECTORY ${output_root})
  nanobind_add_stub(
    shortfin_python_extension_stub_lib_${build_variant}
    MODULE _shortfin_${build_variant}.lib
    OUTPUT ${output_root}/lib/__init__.pyi
    DEPENDS shortfin_python_extension
  )

  nanobind_add_stub(
    shortfin_python_extension_stub_array_${build_variant}
    MODULE _shortfin_${build_variant}.lib.array
    OUTPUT ${output_root}/lib/array.pyi
    DEPENDS shortfin_python_extension
  )

  nanobind_add_stub(
    shortfin_python_extension_stub_local_${build_variant}
    MODULE _shortfin_${build_variant}.lib.local
    OUTPUT ${output_root}/lib/local.pyi
    DEPENDS shortfin_python_extension
  )
endfunction()

if (SHORTFIN_ENABLE_TRACING)
  shortfin_python_stubs(tracy)
else()
  shortfin_python_stubs(default)
endif()
