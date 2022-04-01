# Intel SIMD 文档和资源

## 1. 在线文档

* [Microsoft: VS 2008 Compiler Intrinsics: SSE2 Floating-Point](https://docs.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2008/b88y51e4(v=vs.90))

* [Microsoft: VS 2008 Compiler Intrinsics: SSE3](https://docs.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2008/bb892952(v=vs.90))

* [Microsoft: VS 2010 Compiler Intrinsics: SSE3](https://docs.microsoft.com/en-us/previous-versions/visualstudio/visual-studio-2010/bb892952(v=vs.100))

## 2. 技术文章

* [为什么AVX反而比SSE慢？](https://www.zhihu.com/question/37230675)

    混用 SSE 指令和 AVX 指令将会引发 transition penalty, 但是使用 Intel SIMD intrin 函数写的代码, 是不会出现这个问题的, 除非是手写的 SSE 和 AVX 汇编.

## 3. 在线编译器

* [Godbolt compiler explorer](https://godbolt.org/) [一般可正常访问，可能需要科学上网]

    样例: [Godbolt compiler explorer: clang + gcc + MSVC](https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(j:1,lang:c%2B%2B,source:'%23include+%3Cimmintrin.h%3E%0A%23include+%3Cstdint.h%3E%0A%0Atemplate%3Cunsigned+elem%3E%0Astatic+inline%0A__m256i+merge_epi64(__m256i+v,+int64_t+newval)%0A%7B%0A++++static_assert(elem+%3C%3D+3,+%22a+__m256i+only+has+4+qword+elements%22)%3B%0A++++__m256i+splat+%3D+_mm256_set1_epi64x(newval)%3B%0A%0A++++constexpr+unsigned+dword_blendmask+%3D+0b11+%3C%3C+(elem*2)%3B++//+vpblendd+uses+2+bits+per+qword%0A++++return++_mm256_blend_epi32(v,+splat,+dword_blendmask)%3B%0A%7D%0A%0A%0A//+test+functions%0A__m256i+merge3(__m256i+v,+int64_t+newval)+%7B%0A++++return+merge_epi64%3C3%3E(v,+newval)%3B%0A%7D%0A%0A__m256i+merge2(__m256i+v,+int64_t+newval)+%7B%0A++++return+merge_epi64%3C2%3E(v,+newval)%3B%0A%7D%0A%0A__m256i+merge1(__m256i+v,+int64_t+newval)+%7B%0A++++return+merge_epi64%3C1%3E(v,+newval)%3B%0A%7D%0A%0A__m256i+merge0(__m256i+v,+int64_t+newval)+%7B%0A++++return+merge_epi64%3C0%3E(v,+newval)%3B%0A%7D%0A'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:37.06327601492114,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:clang700,filters:(b:'0',binary:'1',commentOnly:'0',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'1',trim:'1'),fontScale:1.0749542399999998,lang:c%2B%2B,libs:!(),options:'-O3+-Wall+-march%3Dhaswell',source:1),l:'5',n:'0',o:'x86-64+clang+7.0.0+(Editor+%231,+Compiler+%231)+C%2B%2B',t:'0')),k:32.48462940134176,l:'4',m:100,n:'0',o:'',s:0,t:'0'),(g:!((g:!((h:compiler,i:(compiler:gsnapshot,filters:(b:'0',binary:'1',commentOnly:'0',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'1',trim:'1'),fontScale:1.0749542399999998,lang:c%2B%2B,libs:!(),options:'-std%3Dgnu%2B%2B17+-O3+-Wall+-march%3Dhaswell',source:1),l:'5',n:'0',o:'x86-64+gcc+(trunk)+(Editor+%231,+Compiler+%232)+C%2B%2B',t:'0')),header:(),k:30.452094583737093,l:'4',m:50,n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:vcpp_v19_16_x64,filters:(b:'0',binary:'1',commentOnly:'0',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'1',trim:'1'),fontScale:1.0749542399999998,lang:c%2B%2B,libs:!(),options:'-Ox+-arch:AVX2+-Gv',source:1),l:'5',n:'0',o:'x64+msvc+v19.16+(Editor+%231,+Compiler+%233)+C%2B%2B',t:'0')),header:(),l:'4',m:50,n:'0',o:'',s:0,t:'0')),k:30.452094583737093,l:'3',n:'0',o:'',t:'0')),l:'2',n:'0',o:'',t:'0')),version:4)

    ```shell
    clang-x64 参数: -O3 -Wall -march=haswell
    clang-x86 参数: -m32 -O3 -Wall -march=haswell
    gcc-x64 参数:   -std=gnu++17 -O3 -Wall -march=haswell
    gcc-x86 参数:   -m32 -std=gnu++17 -O3 -Wall -march=haswell
    MSVC-x86 参数:  /D "WIN32" /D "NDEBUG" /O2 /Zc:inline /fp:precise /arch:AVX2 /Gv /Gm-
    MSVC-x64 参数:  /D "NDEBUG" /O2 /Zc:inline /fp:precise /arch:AVX2 /Gv /Gm-
    ```

    样例: [Godbolt compiler explorer: clang 9.01 + gcc 11.2](https://www.godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM2akrgAyeAyYAHI%2BAEaYxBKBAA6oCoRODB7evv5JKWkCoeFRLLHxXIF2mA7pQgRMxASZPn4Btpj2jgK19QSFkTFxCbZ1DU3ZrQojvWH9JYPlAJS2qF7EyOwcJhoAguYAzGHI3lgA1CZ7bpP4ggB0COfYW7tmBwxHXqfnbkwKSg13Dye%2B0Ox0wZwueBYLDCBGIYQBe0eOyeBEwLESBlR4LcMJOYSwqkBO0mTEcyDxDFoMyeAH0aSwzABWABseBOUKZzJpYT%2BBBpmESeC4zIgdIZLLZI2AmAIpApBBOADcxF5MAsgQB2Kw7E66k4ksk0n68qD4zCqcGPPYAEROGjOZmZ5mZFIJ2JOwqWDrM2wAagANEAgDks7kMXn8wXCiALECu83sryTE6xM6MiwaAC0XEZJkZ1vMZnVe212z1JzFnLZxEwCi8tAI51LwKoJywNHC6FFNL9/rMdPVOr1eFbEDNFsBNrtDqdjvjE4uJwAHAszlqgS8R3anuXy2LykvJfVpQQD%2BDbfTxVzRJNUpyaakDxApTLi6Xd3r92ZDydaKgAO70ngqiYOgZ7nBeUJhhGApCiKL6nt%2BcrKt4mBymODBul8y6rucABi84nHGGhvjuH4nDWdYNueFYhlyPJxI4B4PngnLPseMpyn%2BgHQiBYFIXapHIi87RKORu5kR%2BAD0UknAAKggcRgiwTAAJ6pngCgnEwKZeMAKZVEwSZggQilKiqYKoK2plgggeDAAgHrfimhBaSp6lgn%2Bkw3JJu5fj%2BCHgVOl73jeBB3qGj7fuxxAnkJZbkf51a1vWiFLjRl7QYxkZwTFJ4HshFnoeO7oruCBElcR8XiZRqUZXRNJhVFS4sWxtUNge8X7K4I6%2BSYGoFkOuqiWCW4YVhiIQcuM7OoR2Gemu76JfSB5snZDlBZBV78qosJMNUzERfBHGyh61XLSwq3ssBoGbbRLBZQ0OXRutCAFeZqHFZhCaTraZX4Sc40JpmOFEYJTa%2BeW7UKlNIWhgxDRCt%2BrUsnlnHXXx71cOdur9YNCXDbQYn9UtH7Gox1BiEoONrvj3VUq27YzF2dK9gOvlJSciQ9AA8lQd1w1y5p7QdyNHWjp1AwuVrYCcewLDTnO8bdzmw1BCN8rB0bcw0fPvShqpfVhTI4RDQ0USl1Fq9tGtIy14sIXKyv8UuRs/ZNiJywrZvPHsPVUL5NYEKsDAW1RjYlpqdM7DJJyosmVBeG8nThrS9JVhSvIaN2V5soqcq4gbaqLYHMoh%2ByNvhtlWtOhcGgPBA%2BcfaqXUDUCOyVhKmeMZIOcZ03hcWbh67m0H5cNRrz2124kgN03Ret9H2yd6y3cNEufddwPgjN8XJOl8HxChxPVdPTXXxLnPhWoYv7fL%2BnXca%2BUm%2Br9vCoLyXo9l0fFf3pP58XHKFfXet8dgcCWLQTgjJeB%2BA4FoUgqBOBuGsNYfUKw1hgn2DwUgBBNDgKWAAaxAHsPYNwSHkIoZQ5k%2BhOCSBgXghBnBeAKBABoHBeClhwFgEgNA6I6BxHIJQXhgp6DxCOIYYAABODQXA%2BB0FRMQFhEBogMOiGEeoqlODYN4WwQQPNKSaLgbwLAKkjDiCMaQfANZqiKlrAw80VQvCoi0bwGE7QGFUmiMQDRHgsAuJwXCFgLilhUAMMABQvo8CYH/DzRIjB/H8EECIMQ7ApAyEEIoFQ6gLG6FkQYIwKAUGWH0HgaILDIBLFQIkFOLCOCZh5nsE4mYADqYhaBNJUmse4NoFAENUgYAhmBmHtCqCnFwmExh%2BFkSEGYxRSh6GSKkFOkyFl5BTn0OZ8w2gdBqFMFZsjKjVC6FMDZAwyjDB6Psi5DRTlzDKEsBQ6D1h6FhJgDYPAIFQPoRYxBHBVBLmZJmZkkgTjiKMCcSRNwNA3C4IDZBlhrBylwIQEgDo9iyJOB4Phoi0XY14LgoxCtSBEJIWQyh5KSHUMgRwOhpBYHwN%2Bcw1h7DCWkC4YgFAqBsUCIoBAYR/D4jAC4OUORDY4hKJURYtRzBiCGO0Vy3RBB9G0EMfAkxEjzFqrwNYxwtjanwIccgJx7zXGCHcRYzx3jZW%2BJNQEyEwS%2BBhIiVEmJcTYHYMScIUQ4g0mesyWoBhuhAj5OMEUmwnjykxgQdU9ItTMxXAgsABgXgw1cA1E0hpTTWm0HaZmTpyBunWl6f0pggzhk7OcBAVwVyZlFDOaspZ6QrmLPyAwW58yDkjKOQwbooxPDND0IclOvbph1ruYOvZ/bsgHJObM%2Bt2NlirGebI15JrPkcGgXShhvz/mAuBScYAyByTCpuGYOFYakX4CIMQXFcosUiLiGios%2BKOGEOIaQilFKaE0u%2BQyphthmUEq0ES6lZhf28EZSy4DSxbGKPSCASQQA%3D)

    ```shell
    clang 参数: -O3 -Wall -march=skylake
    gcc 参数:   -std=gnu++17 -O3 -Wall -march=skylake
    MSVC-x86 参数:  /D "WIN32" /D "NDEBUG" /O2 /Zc:inline /fp:precise /arch:AVX2 /Gv /Gm-
    MSVC-x64 参数:  /D "NDEBUG" /O2 /Zc:inline /fp:precise /arch:AVX2 /Gv /Gm-
    MSVC 参数:  -Ox -arch:AVX2 -Gv
    ```

    参数 `'-march=icelake'` 可能绝大部分编译器未支持.

## 4. 参考文章

* [StackoverFlow: Move an int64_t to the high quadwords of an AVX2 __m256i vector](https://stackoverflow.com/questions/54048226/move-an-int64-t-to-the-high-quadwords-of-an-avx2-m256i-vector)
