# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.28.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.28.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/manujayadharan/git_repos/XXXDD

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/manujayadharan/git_repos/XXXDD

# Include any dependencies generated for this target.
include CMakeFiles/ElastDD.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ElastDD.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ElastDD.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ElastDD.dir/flags.make

CMakeFiles/ElastDD.dir/src/elast_dd.cc.o: CMakeFiles/ElastDD.dir/flags.make
CMakeFiles/ElastDD.dir/src/elast_dd.cc.o: src/elast_dd.cc
CMakeFiles/ElastDD.dir/src/elast_dd.cc.o: CMakeFiles/ElastDD.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/manujayadharan/git_repos/XXXDD/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ElastDD.dir/src/elast_dd.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ElastDD.dir/src/elast_dd.cc.o -MF CMakeFiles/ElastDD.dir/src/elast_dd.cc.o.d -o CMakeFiles/ElastDD.dir/src/elast_dd.cc.o -c /Users/manujayadharan/git_repos/XXXDD/src/elast_dd.cc

CMakeFiles/ElastDD.dir/src/elast_dd.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/ElastDD.dir/src/elast_dd.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/manujayadharan/git_repos/XXXDD/src/elast_dd.cc > CMakeFiles/ElastDD.dir/src/elast_dd.cc.i

CMakeFiles/ElastDD.dir/src/elast_dd.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/ElastDD.dir/src/elast_dd.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/manujayadharan/git_repos/XXXDD/src/elast_dd.cc -o CMakeFiles/ElastDD.dir/src/elast_dd.cc.s

CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.o: CMakeFiles/ElastDD.dir/flags.make
CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.o: src/elasticity_mfedd.cc
CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.o: CMakeFiles/ElastDD.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/manujayadharan/git_repos/XXXDD/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.o -MF CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.o.d -o CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.o -c /Users/manujayadharan/git_repos/XXXDD/src/elasticity_mfedd.cc

CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/manujayadharan/git_repos/XXXDD/src/elasticity_mfedd.cc > CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.i

CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/manujayadharan/git_repos/XXXDD/src/elasticity_mfedd.cc -o CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.s

# Object files for target ElastDD
ElastDD_OBJECTS = \
"CMakeFiles/ElastDD.dir/src/elast_dd.cc.o" \
"CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.o"

# External object files for target ElastDD
ElastDD_EXTERNAL_OBJECTS =

ElastDD: CMakeFiles/ElastDD.dir/src/elast_dd.cc.o
ElastDD: CMakeFiles/ElastDD.dir/src/elasticity_mfedd.cc.o
ElastDD: CMakeFiles/ElastDD.dir/build.make
ElastDD: /Users/manujayadharan/Downloads/dealii-9.5.2/build/lib/libdeal_II.g.9.5.2.dylib
ElastDD: /opt/homebrew/opt/openmpi/lib/libmpi_usempif08.dylib
ElastDD: /opt/homebrew/opt/openmpi/lib/libmpi_usempi_ignore_tkr.dylib
ElastDD: /opt/homebrew/opt/openmpi/lib/libmpi_mpifh.dylib
ElastDD: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.2.sdk/usr/lib/libz.tbd
ElastDD: /opt/homebrew/lib/libboost_iostreams-mt.dylib
ElastDD: /opt/homebrew/lib/libboost_serialization-mt.dylib
ElastDD: /opt/homebrew/lib/libboost_system-mt.dylib
ElastDD: /opt/homebrew/lib/libboost_thread-mt.dylib
ElastDD: /opt/homebrew/lib/libboost_regex-mt.dylib
ElastDD: /opt/homebrew/lib/libboost_chrono-mt.dylib
ElastDD: /opt/homebrew/lib/libboost_atomic-mt.dylib
ElastDD: /opt/homebrew/lib/libmetis.dylib
ElastDD: /opt/homebrew/opt/openmpi/lib/libmpi.dylib
ElastDD: CMakeFiles/ElastDD.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/manujayadharan/git_repos/XXXDD/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ElastDD"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ElastDD.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ElastDD.dir/build: ElastDD
.PHONY : CMakeFiles/ElastDD.dir/build

CMakeFiles/ElastDD.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ElastDD.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ElastDD.dir/clean

CMakeFiles/ElastDD.dir/depend:
	cd /Users/manujayadharan/git_repos/XXXDD && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/manujayadharan/git_repos/XXXDD /Users/manujayadharan/git_repos/XXXDD /Users/manujayadharan/git_repos/XXXDD /Users/manujayadharan/git_repos/XXXDD /Users/manujayadharan/git_repos/XXXDD/CMakeFiles/ElastDD.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/ElastDD.dir/depend

