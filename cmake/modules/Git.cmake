function(git_get_head_revision _refspecvar _hashvar)
execute_process(COMMAND git log --pretty=format:'%h' -n 1
WORKING_DIRECTORY
"${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE GIT_REV
                ERROR_QUIET)

                message(STATUS "${CMAKE_CURRENT_SOURCE_DIR} -> ${GIT_REV}")

# Check whether we got any revision (which isn't
# always the case, e.g. when someone downloaded a zip
# file from Github instead of a checkout)
if ("${GIT_REV}" STREQUAL "")
    set(GIT_REV "N/A")
    set(GIT_DIFF "")
    set(GIT_TAG "N/A")
    set(GIT_BRANCH "N/A")
else()
    execute_process(
        COMMAND bash -c "git diff --quiet --exit-code || echo +"
        WORKING_DIRECTORY
        "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_DIFF)

        
        execute_process(
            COMMAND git describe --exact-match --tags
            WORKING_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE GIT_TAG ERROR_QUIET)
            
            execute_process(
                COMMAND git rev-parse --abbrev-ref HEAD
                WORKING_DIRECTORY
                "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE GIT_BRANCH)
                
                string(STRIP "${GIT_REV}" GIT_REV)
                string(SUBSTRING "${GIT_REV}" 1 7 GIT_REV)
                string(STRIP "${GIT_DIFF}" GIT_DIFF)
                string(STRIP "${GIT_TAG}" GIT_TAG)
                string(STRIP "${GIT_BRANCH}" GIT_BRANCH)
    # message(STATUS "${CMAKE_CURRENT_SOURCE_DIR} -> ${GIT_DIFF}")
    # message(STATUS "${CMAKE_CURRENT_SOURCE_DIR} -> ${GIT_TAG}")
    # message(STATUS "${CMAKE_CURRENT_SOURCE_DIR} -> ${GIT_BRANCH}")

    if (NOT "${GIT_TAG}" STREQUAL "")
    set(${_refspecvar} ${GIT_TAG} PARENT_SCOPE)
    elseif(NOT "${GIT_BRANCH}" STREQUAL "")
    set(${_refspecvar} ${GIT_BRANCH} PARENT_SCOPE)
    else()
    set(${_refspecvar} "?" PARENT_SCOPE)
    endif()
    set(${_hashvar} ${GIT_REV} PARENT_SCOPE)
endif()

endfunction()


# returns "" if clean or "+" if dirty
function(git_local_changes _var)
execute_process(COMMAND
		git
		diff-index --quiet HEAD --
		WORKING_DIRECTORY
		"${CMAKE_CURRENT_SOURCE_DIR}"
		RESULT_VARIABLE
		res
		OUTPUT_VARIABLE
		out
		ERROR_QUIET
		OUTPUT_STRIP_TRAILING_WHITESPACE)
	if(res EQUAL 0)
		set(${_var} "CLEAN" PARENT_SCOPE)
	else()
		set(${_var} "DIRTY" PARENT_SCOPE)
    endif()
endfunction()

