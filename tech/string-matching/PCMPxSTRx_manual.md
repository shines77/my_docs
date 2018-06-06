
# Intel SSE 4.2 PCMPxSTRx 指令用户手册

* `X86 Assembly/SSE`

    [https://en.wikibooks.org/wiki/X86_Assembly/SSE]()

    关于 `pcmpistri`，`pcmpistrm`，`pcmpestri`，`pcmpestrm` 指令的详细说明等。

    * `pcmpistri` 指令

    ```c
    Intel syntax:

        pcmpistri arg2, arg1, IMM8

        PCMPISTRI, Packed Compare Implicit Length Strings, Return Index. Compares strings of implicit length and generates index in ECX.

    Operands

    arg1:
        * XMM Register
        * Memory

    arg2:
        * XMM Register

    IMM8:
        * 8-bit Immediate value

    Modified flags

        1. CF is reset if IntRes2 is zero, set otherwise
        2. ZF is set if a null terminating character is found in arg2, reset otherwise
        3. SF is set if a null terminating character is found in arg1, reset otherwise
        4. OF is set to IntRes2[0]
        5. AF is reset
        6. PF is reset
    ```

    * `pcmpistrm` 指令

    ```c
    Intel syntax:

        pcmpistrm arg2, arg1, IMM8

        PCMPISTRM, Packed Compare Implicit Length Strings, Return Mask. Compares strings of implicit length and generates a mask stored in XMM0.

    Operands

    arg1:
        * XMM Register

    arg2:
        * XMM Register
        * Memory

    IMM8:
        * 8-bit Immediate value

    Modified flags

        1. CF is reset if IntRes2 is zero, set otherwise
        2. ZF is set if a null terminating character is found in arg2, reset otherwise
        3. SF is set if a null terminating character is found in arg2, reset otherwise
        4. OF is set to IntRes2[0]
        5. AF is reset
        6. PF is reset
    ```

    * `pcmpestri` 指令

    ```c
    Intel syntax:

        pcmpestri arg2, arg1, IMM8

        PCMPESTRI, Packed Compare Explicit Length Strings, Return Index. Compares strings of explicit length and generates index in ECX.

    Operands

    arg1:
        * XMM Register

    arg2:
        * XMM Register
        * Memory

    IMM8:
        * 8-bit Immediate value

    Implicit Operands

        * EAX holds the length of arg1
        * EDX holds the length of arg2

    Modified flags

        1. CF is reset if IntRes2 is zero, set otherwise
        2. ZF is set if EDX is < 16(for bytes) or 8(for words), reset otherwise
        3. SF is set if EAX is < 16(for bytes) or 8(for words), reset otherwise
        4. OF is set to IntRes2[0]
        5. AF is reset
        6. PF is reset
    ```

    * `pcmpestrm` 指令

    ```c
    Intel syntax:

        pcmpestrm arg2, arg1, IMM8

        PCMPESTRM, Packed Compare Explicit Length Strings, Return Mask. Compares strings of explicit length and generates a mask stored in XMM0.

    Operands

    arg1:
        * XMM Register

    arg2:
        * XMM Register
        * Memory

    IMM8:
        * 8-bit Immediate value

    Implicit Operands

        * EAX holds the length of arg1
        * EDX holds the length of arg2

    Modified flags

        1. CF is reset if IntRes2 is zero, set otherwise
        2. ZF is set if EDX is < 16(for bytes) or 8(for words), reset otherwise
        3. ZSF is set if EAX is < 16(for bytes) or 8(for words), reset otherwise
        4. ZOF is set to IntRes2[0]
        5. ZAF is reset
        6. ZPF is reset
    ```
