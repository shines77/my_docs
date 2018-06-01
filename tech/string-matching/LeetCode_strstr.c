
int strStr(char * haystack_start, char * needle_start) {
    const char * haystack = haystack_start;
    const char * needle = needle_start;
    if (*needle != '\0') {
        const char * scan_start = haystack;
        while (*scan_start != '\0') {
            const char * cursor = needle;
            while (*scan_start != *cursor) {
                scan_start++;
                if (*scan_start == '\0')
                    return -1;
            }

            const char * scan = scan_start;
            do {
                cursor++;
                scan++;
                if (*cursor != '\0') {
                    if (*scan != '\0') {
                        if (*scan != *cursor) {
                            scan_start++;
                            break;
                        }
                    }
                    else {
                        return -1;
                    }
                }
                else {
                    return (int)(scan_start - haystack_start);
                }
            } while (1);
        }
        return -1;
    }
    else {
        return 0;
    }
}

int strStr(char * haystack, char * needle) {
    int i, j;
    if (needle[0] == '\0')
        return 0;
    for (i = 0; haystack[i] != '\0'; i++) {
        for (j = 0; haystack[i + j] == needle[j]; j++) {
            if (needle[j + 1] == '\0')
                return i;
            if (haystack[i + j + 1] == '\0')
                return -1;
        }
    }
    return -1;
}
