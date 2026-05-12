import sys
import re

def instrument_execution(codegen_path):
    # Read file
    with open(codegen_path + "/Core0.c", "r") as f:
        code = f.read()

    # --------------------------------------------------------------------
    # 1. Insert timespec subtraction function after "// Core Global Definitions"
    # --------------------------------------------------------------------
    global_insert = '''
    enum { NS_PER_SECOND = 1000000000 };
    void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
    {
        td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
        td->tv_sec  = t2.tv_sec - t1.tv_sec;
        if (td->tv_sec > 0 && td->tv_nsec < 0)
        {
            td->tv_nsec += NS_PER_SECOND;
            td->tv_sec--;
        }
        else if (td->tv_sec < 0 && td->tv_nsec > 0)
        {
            td->tv_nsec -= NS_PER_SECOND;
            td->tv_sec++;
        }
    }
    '''

    code = code.replace("// Core Global Definitions",
                        global_insert + "\n// Core Global Definitions")

    # --------------------------------------------------------------------
    # 2. Insert variable declarations before the 'for(index...' line
    # --------------------------------------------------------------------
    decl_insert = 'struct timespec start, finish, delta, latence = {0,0};\n'

    code = re.sub(
        r'(?=for\s*\(\s*index\s*=\s*0\s*;)',
        decl_insert,
        code,
        count=1
    )

    # --------------------------------------------------------------------
    # NEW STEP — Insert clock_gettime(CLOCK_REALTIME, &start) after "// loop body"
    # --------------------------------------------------------------------

    body_insert = 'clock_gettime(CLOCK_REALTIME, &start);\n'

    code = code.replace(
        "// loop body",
        "// loop body\n    " + body_insert    # keep indentation
    )

    # --------------------------------------------------------------------
    # 3. Insert inside loop after "// loop footer\n    pthread_barrier_wait(&iter_barrier);"
    # --------------------------------------------------------------------
    loop_footer_pattern = (
        r'//\s*loop footer\s*\n'          # comment line
        r'[ \t]*pthread_barrier_wait\s*'  # indentation + function name
        r'\(\s*&iter_barrier\s*\)\s*;'    # parentheses and semicolon
    )

    loop_insert = '''
        clock_gettime(CLOCK_REALTIME, &finish);
        sub_timespec(start, finish, &delta);
        latence.tv_nsec += delta.tv_nsec;
        latence.tv_sec += delta.tv_sec;
    '''

    code = re.sub(loop_footer_pattern,
                  lambda m: m.group(0) + loop_insert,
                  code,
                  count=1)

    # --------------------------------------------------------------------
    # 4. Insert average print after the closing brace of the loop
    # --------------------------------------------------------------------

    avg_insert = '''
    printf("latency : %d.%.9ld s\\n",
           (int) latence.tv_sec / PREESM_LOOP_SIZE,
           latence.tv_nsec / PREESM_LOOP_SIZE);
    '''

    # Find loop footer occurrence
    footer_match = re.search(loop_footer_pattern, code)
    if footer_match:
        footer_end = footer_match.end()

        # Find the next closing brace after the footer
        closing_brace_pos = code.find('}', footer_end)

        if closing_brace_pos != -1:
            # Insert the average print AFTER this brace
            code = (code[:closing_brace_pos+1] +
                    avg_insert +
                    code[closing_brace_pos+1:])
        else:
            print("Warning: closing brace after loop footer not found.")
    else:
        print("Warning: loop footer not found. rololoooo")

    # --------------------------------------------------------------------
    # Write result back to file
    # --------------------------------------------------------------------
    with open(codegen_path + "/Core0.c", "w") as f:
        f.write(code)


    # write loop number in preesm_gen.h
    define_line = "#define PREESM_LOOP_SIZE 1\n"
    try:
        with open(codegen_path + "/preesm_gen.h", "r") as f:
            lines = f.readlines()
    except IOError:
        print(f"Error: cannot read preesm_gen.h")
        sys.exit(1)

    new_lines = []
    inserted = False

    for line in lines:
        if not inserted and line.lstrip().startswith("#ifdef PREESM_LOOP_SIZE"):
            # Insert BEFORE the #ifdef line
            new_lines.append(define_line)
            inserted = True
        new_lines.append(line)

    try:
        with open(codegen_path + "/preesm_gen.h", "w") as f:
            f.writelines(new_lines)
        print(f"Inserted PREESM_LOOP_SIZE define in: preesm_gen.h")
    except IOError:
        print(f"Error: cannot write preesm_gen.h")

    print(f"Instrumentation done")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python instrument_latency.py <file.cpp>")
        sys.exit(1)

    filename = sys.argv[1]
    instrument_execution(filename)