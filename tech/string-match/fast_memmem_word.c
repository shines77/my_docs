
//
// See: https://blog.csdn.net/liangzhao_jay/article/details/8792486
//
int fast_memmem(const char * text, int text_len, const char * pattern, int pattern_len) {
    char used[256];
    int i, tpos, ppos, match, half_p_len;

    for (i = 0; i < 256; i++) {
        used[i] = 0;
    }

    // Preprocessing
    half_p_len = pattern_len / 2;
    for (i = 0; i < pattern_len; i++) {
        used[pattern[i]] = 1;
    }

    // Searching
    for (i = 0; i <= (text_len – pattern_len); i++) {
        match = 1;
        tpos = i + pattern_len – 1;
        ppos = pattern_len – 1;
        for (; ppos >= 0; --tpos, --ppos) {
            if (used[text[tpos]] == 0) {
                i = tpos;
                match = 0;
                break;
            }
            if (match && text[tpos] != pattern[ppos]) {
                match = 0;
                if (ppos > half_p_len)
                    break;
            }
        }
        if (match != 0) {
            return i;
        }
    }

    return -1;
}

int fast_memmem_word(const char * text, int text_len, const char * pattern, int pattern_len) {
    char used[65536];
    int i, tpos, ppos, match, half_p_len;
    unsigned short word;

    for (i = 0; i < 256; i++) {
        used[i] = 0;
    }

    // Preprocessing
    half_p_len = pattern_len / 2;
    for (i = 0; i < pattern_len - 1; i++) {
        word = *(unsigned short *)&pattern[i];
        used[word] = 1;
    }

    // Searching
    for (i = 0; i <= (text_len – pattern_len); i++) {
        match = 1;
        tpos = i + pattern_len – 1;
        ppos = pattern_len – 1;
        for (; ppos >= 0; --tpos, --ppos) {
            word = *(unsigned short *)&text[tpos];
            if (used[word] == 0) {
                i = tpos + 1;
                match = 0;
                break;
            }
            if (match && text[tpos] != pattern[ppos]) {
                match = 0;
                if (ppos > half_p_len)
                    break;
            }
        }
        if (match != 0) {
            return i;
        }
    }

    return -1;
}
