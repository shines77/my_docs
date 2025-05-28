
# 64bit rANS Encoder and Decoder

## 问题

请用 C++ 写一个 rANS 压缩算法的 encoder 和 decoder，使用 64 位整型版本，使用动态freq估算，并添加英文注释，不使用C++标准库STL，输入输出请使用自定义的二进制stream或buffer类。

## 回答

（来自 Trae-v1.3.7 - Claude-3.7-Sonnet）

### rANS 压缩算法实现

我将为您实现一个完整的 rANS (range Asymmetric Numeral Systems) 压缩算法，包括编码器和解码器。这个实现将使用 64 位整型，包含动态频率估算，并且不使用 C++ 标准库。

#### 实现概述

我将创建以下文件：

1. 自定义二进制流类 (`BitStream.h`)
2. rANS 编码器和解码器 (`rANS.h` 和 `rANS.cpp`)
3. 一个简单的示例程序 (`main.cpp`)

让我们开始实现：

`\compression\ZipLab\BitStream.h`：

```cpp
#ifndef BIT_STREAM_H
#define BIT_STREAM_H

// Custom binary stream class for rANS compression
// Provides bit-level and byte-level operations for I/O
class BitStream {
private:
    unsigned char* buffer;    // Internal buffer to store data
    unsigned int bufferSize;  // Size of the buffer in bytes
    unsigned int position;    // Current position in the buffer (in bytes)
    unsigned int bitPosition; // Current bit position within the current byte (0-7)
    bool isReading;           // Flag to indicate if stream is in read mode

public:
    // Constructor - initialize an empty stream for writing
    BitStream(unsigned int initialSize = 1024) {
        buffer = new unsigned char[initialSize];
        bufferSize = initialSize;
        position = 0;
        bitPosition = 0;
        isReading = false;

        // Initialize buffer to zeros
        for (unsigned int i = 0; i < bufferSize; i++) {
            buffer[i] = 0;
        }
    }

    // Constructor - initialize a stream for reading from existing data
    BitStream(unsigned char* data, unsigned int size) {
        buffer = new unsigned char[size];
        bufferSize = size;
        position = 0;
        bitPosition = 0;
        isReading = true;

        // Copy data to internal buffer
        for (unsigned int i = 0; i < size; i++) {
            buffer[i] = data[i];
        }
    }

    // Destructor
    ~BitStream() {
        delete[] buffer;
    }

    // Write a single bit to the stream
    void WriteBit(bool bit) {
        if (isReading) return; // Cannot write in read mode

        // Ensure we have enough space
        if (position >= bufferSize) {
            Resize(bufferSize * 2);
        }

        // Set or clear the bit at the current position
        if (bit) {
            buffer[position] |= (1 << (7 - bitPosition));
        } else {
            buffer[position] &= ~(1 << (7 - bitPosition));
        }

        // Move to the next bit
        bitPosition++;
        if (bitPosition == 8) {
            bitPosition = 0;
            position++;
        }
    }

    // Read a single bit from the stream
    bool ReadBit() {
        if (!isReading || position >= bufferSize) return false;

        // Get the bit at the current position
        bool bit = (buffer[position] & (1 << (7 - bitPosition))) != 0;

        // Move to the next bit
        bitPosition++;
        if (bitPosition == 8) {
            bitPosition = 0;
            position++;
        }

        return bit;
    }

    // Write a byte to the stream
    void WriteByte(unsigned char byte) {
        if (isReading) return; // Cannot write in read mode

        // If we're not aligned to a byte boundary, write bit by bit
        if (bitPosition != 0) {
            for (int i = 7; i >= 0; i--) {
                WriteBit((byte & (1 << i)) != 0);
            }
            return;
        }

        // Ensure we have enough space
        if (position >= bufferSize) {
            Resize(bufferSize * 2);
        }

        // Write the byte directly
        buffer[position++] = byte;
    }

    // Read a byte from the stream
    unsigned char ReadByte() {
        if (!isReading || position >= bufferSize) return 0;

        // If we're not aligned to a byte boundary, read bit by bit
        if (bitPosition != 0) {
            unsigned char byte = 0;
            for (int i = 7; i >= 0; i--) {
                if (ReadBit()) {
                    byte |= (1 << i);
                }
            }
            return byte;
        }

        // Read the byte directly
        return buffer[position++];
    }

    // Write multiple bytes to the stream
    void WriteBytes(const unsigned char* data, unsigned int size) {
        for (unsigned int i = 0; i < size; i++) {
            WriteByte(data[i]);
        }
    }

    // Read multiple bytes from the stream
    void ReadBytes(unsigned char* data, unsigned int size) {
        for (unsigned int i = 0; i < size; i++) {
            data[i] = ReadByte();
        }
    }

    // Write a 32-bit integer to the stream
    void WriteInt32(unsigned int value) {
        WriteByte((value >> 24) & 0xFF);
        WriteByte((value >> 16) & 0xFF);
        WriteByte((value >> 8) & 0xFF);
        WriteByte(value & 0xFF);
    }

    // Read a 32-bit integer from the stream
    unsigned int ReadInt32() {
        unsigned int value = 0;
        value = (value << 8) | ReadByte();
        value = (value << 8) | ReadByte();
        value = (value << 8) | ReadByte();
        value = (value << 8) | ReadByte();
        return value;
    }

    // Write a 64-bit integer to the stream
    void WriteInt64(unsigned long long value) {
        WriteInt32((unsigned int)((value >> 32) & 0xFFFFFFFF));
        WriteInt32((unsigned int)(value & 0xFFFFFFFF));
    }

    // Read a 64-bit integer from the stream
    unsigned long long ReadInt64() {
        unsigned long long value = 0;
        value = ((unsigned long long)ReadInt32()) << 32;
        value |= ReadInt32();
        return value;
    }

    // Get the current size of the data in the stream (in bytes)
    unsigned int GetSize() {
        return position + (bitPosition > 0 ? 1 : 0);
    }

    // Get a pointer to the internal buffer
    unsigned char* GetBuffer() {
        return buffer;
    }

    // Align to the next byte boundary (for writing)
    void AlignToByte() {
        if (bitPosition > 0) {
            bitPosition = 0;
            position++;
        }
    }

private:
    // Resize the internal buffer
    void Resize(unsigned int newSize) {
        unsigned char* newBuffer = new unsigned char[newSize];

        // Copy existing data
        for (unsigned int i = 0; i < bufferSize; i++) {
            newBuffer[i] = buffer[i];
        }

        // Initialize new space to zeros
        for (unsigned int i = bufferSize; i < newSize; i++) {
            newBuffer[i] = 0;
        }

        // Replace the old buffer
        delete[] buffer;
        buffer = newBuffer;
        bufferSize = newSize;
    }
};

#endif // BIT_STREAM_H
```

`\compression\ZipLab\rANS.h`：

```cpp
#ifndef RANS_H
#define RANS_H

#include "BitStream.h"

// Constants for rANS algorithm
#define PROB_BITS 14                    // Number of bits for probability representation
#define PROB_SCALE (1 << PROB_BITS)     // Scale for probability (16384)
#define STATE_BITS 32                   // Number of bits for state
#define RANS_L ((unsigned long long)1 << STATE_BITS)  // Lower bound for renormalization

// Maximum number of symbols (typically 256 for byte-oriented compression)
#define MAX_SYMBOLS 256

// rANS encoder class
class RansEncoder {
private:
    unsigned long long state;           // Current encoder state
    unsigned int frequencies[MAX_SYMBOLS]; // Frequency table
    unsigned int cumFrequencies[MAX_SYMBOLS + 1]; // Cumulative frequency table
    unsigned int totalSymbols;          // Total number of symbols processed
    BitStream* outputStream;            // Output stream for compressed data

public:
    // Constructor
    RansEncoder(BitStream* output);

    // Initialize the encoder
    void Initialize();

    // Update frequency tables with a new symbol
    void UpdateFrequencies(unsigned char symbol);

    // Encode a symbol
    void Encode(unsigned char symbol);

    // Finalize encoding and write any remaining data
    void Finalize();

    // Write frequency table to the output stream
    void WriteFrequencyTable();

private:
    // Normalize frequencies to fit within PROB_SCALE
    void NormalizeFrequencies();

    // Build cumulative frequency table from frequencies
    void BuildCumulativeFrequencies();
};

// rANS decoder class
class RansDecoder {
private:
    unsigned long long state;           // Current decoder state
    unsigned int frequencies[MAX_SYMBOLS]; // Frequency table
    unsigned int cumFrequencies[MAX_SYMBOLS + 1]; // Cumulative frequency table
    BitStream* inputStream;             // Input stream for compressed data

public:
    // Constructor
    RansDecoder(BitStream* input);

    // Initialize the decoder
    void Initialize();

    // Decode a symbol
    unsigned char Decode();

    // Read frequency table from the input stream
    void ReadFrequencyTable();

private:
    // Build cumulative frequency table from frequencies
    void BuildCumulativeFrequencies();

    // Find symbol based on cumulative frequency
    unsigned char FindSymbol(unsigned int cumFreq);
};

#endif // RANS_H
```

`\compression\ZipLab\rANS.cpp`：

```cpp
#include "rANS.h"

// RansEncoder implementation

RansEncoder::RansEncoder(BitStream* output) {
    outputStream = output;
    Initialize();
}

void RansEncoder::Initialize() {
    // Initialize state to RANS_L
    state = RANS_L;

    // Initialize frequency table with 1s (to avoid zero probabilities)
    for (int i = 0; i < MAX_SYMBOLS; i++) {
        frequencies[i] = 1;
    }

    // Initialize total symbols count
    totalSymbols = MAX_SYMBOLS;

    // Build initial cumulative frequency table
    BuildCumulativeFrequencies();
}

void RansEncoder::UpdateFrequencies(unsigned char symbol) {
    // Increment frequency for the symbol
    frequencies[symbol]++;
    totalSymbols++;

    // Periodically normalize frequencies to prevent overflow
    // and adapt to changing statistics in the data
    if (totalSymbols >= PROB_SCALE) {
        NormalizeFrequencies();
        BuildCumulativeFrequencies();
    }
}

void RansEncoder::NormalizeFrequencies() {
    unsigned int total = 0;

    // Scale down frequencies while maintaining their ratios
    for (int i = 0; i < MAX_SYMBOLS; i++) {
        // Scale down by half, but ensure at least 1 for each symbol
        frequencies[i] = (frequencies[i] + 1) / 2;
        total += frequencies[i];
    }

    // Adjust to ensure total is exactly PROB_SCALE
    if (total > PROB_SCALE) {
        // Find symbols with highest frequencies and reduce them
        while (total > PROB_SCALE) {
            unsigned int maxFreq = 0;
            int maxIndex = 0;

            for (int i = 0; i < MAX_SYMBOLS; i++) {
                if (frequencies[i] > maxFreq) {
                    maxFreq = frequencies[i];
                    maxIndex = i;
                }
            }

            frequencies[maxIndex]--;
            total--;
        }
    } else if (total < PROB_SCALE) {
        // Find symbols with lowest frequencies and increase them
        while (total < PROB_SCALE) {
            unsigned int minFreq = PROB_SCALE;
            int minIndex = 0;

            for (int i = 0; i < MAX_SYMBOLS; i++) {
                if (frequencies[i] < minFreq) {
                    minFreq = frequencies[i];
                    minIndex = i;
                }
            }

            frequencies[minIndex]++;
            total++;
        }
    }

    totalSymbols = PROB_SCALE;
}

void RansEncoder::BuildCumulativeFrequencies() {
    cumFrequencies[0] = 0;

    for (int i = 0; i < MAX_SYMBOLS; i++) {
        cumFrequencies[i + 1] = cumFrequencies[i] + frequencies[i];
    }
}

void RansEncoder::Encode(unsigned char symbol) {
    // Get symbol frequency and cumulative frequency
    unsigned int freq = frequencies[symbol];
    unsigned int cumFreq = cumFrequencies[symbol];

    // rANS encoding step
    unsigned long long quotient = state / freq;
    unsigned long long remainder = state % freq;
    state = quotient * PROB_SCALE + remainder + cumFreq;

    // Renormalization
    while (state >= RANS_L) {
        outputStream->WriteByte(state & 0xFF);
        state >>= 8;
    }

    // Update frequency table for adaptive compression
    UpdateFrequencies(symbol);
}

void RansEncoder::WriteFrequencyTable() {
    // Write the frequency table to the output stream
    for (int i = 0; i < MAX_SYMBOLS; i++) {
        outputStream->WriteInt32(frequencies[i]);
    }
}

void RansEncoder::Finalize() {
    // Write the final state
    outputStream->WriteInt64(state);

    // Write the frequency table
    WriteFrequencyTable();
}

// RansDecoder implementation

RansDecoder::RansDecoder(BitStream* input) {
    inputStream = input;
}

void RansDecoder::Initialize() {
    // Read the frequency table
    ReadFrequencyTable();

    // Build cumulative frequency table
    BuildCumulativeFrequencies();

    // Read the initial state
    state = inputStream->ReadInt64();
}

void RansDecoder::ReadFrequencyTable() {
    // Read the frequency table from the input stream
    for (int i = 0; i < MAX_SYMBOLS; i++) {
        frequencies[i] = inputStream->ReadInt32();
    }
}

void RansDecoder::BuildCumulativeFrequencies() {
    cumFrequencies[0] = 0;

    for (int i = 0; i < MAX_SYMBOLS; i++) {
        cumFrequencies[i + 1] = cumFrequencies[i] + frequencies[i];
    }
}

unsigned char RansDecoder::FindSymbol(unsigned int cumFreq) {
    // Binary search to find the symbol based on cumulative frequency
    int left = 0;
    int right = MAX_SYMBOLS;

    while (left + 1 < right) {
        int mid = (left + right) / 2;

        if (cumFrequencies[mid] <= cumFreq) {
            left = mid;
        } else {
            right = mid;
        }
    }

    return (unsigned char)left;
}

unsigned char RansDecoder::Decode() {
    // Extract the cumulative frequency from the state
    unsigned int cumFreq = state % PROB_SCALE;

    // Find the corresponding symbol
    unsigned char symbol = FindSymbol(cumFreq);

    // Get symbol frequency and cumulative frequency
    unsigned int freq = frequencies[symbol];
    unsigned int symCumFreq = cumFrequencies[symbol];

    // Update state
    state = freq * (state / PROB_SCALE) + (cumFreq - symCumFreq);

    // Renormalization
    while (state < RANS_L) {
        state = (state << 8) | inputStream->ReadByte();
    }

    // Update frequency table for adaptive decompression
    frequencies[symbol]++;

    // Rebuild cumulative frequencies periodically
    // This should match the encoder's normalization frequency
    unsigned int totalSymbols = 0;
    for (int i = 0; i < MAX_SYMBOLS; i++) {
        totalSymbols += frequencies[i];
    }

    if (totalSymbols >= PROB_SCALE) {
        // Scale down frequencies
        totalSymbols = 0;
        for (int i = 0; i < MAX_SYMBOLS; i++) {
            frequencies[i] = (frequencies[i] + 1) / 2;
            totalSymbols += frequencies[i];
        }

        // Adjust to ensure total is exactly PROB_SCALE
        if (totalSymbols > PROB_SCALE) {
            while (totalSymbols > PROB_SCALE) {
                unsigned int maxFreq = 0;
                int maxIndex = 0;

                for (int i = 0; i < MAX_SYMBOLS; i++) {
                    if (frequencies[i] > maxFreq) {
                        maxFreq = frequencies[i];
                        maxIndex = i;
                    }
                }

                frequencies[maxIndex]--;
                totalSymbols--;
            }
        } else if (totalSymbols < PROB_SCALE) {
            while (totalSymbols < PROB_SCALE) {
                unsigned int minFreq = PROB_SCALE;
                int minIndex = 0;

                for (int i = 0; i < MAX_SYMBOLS; i++) {
                    if (frequencies[i] < minFreq) {
                        minFreq = frequencies[i];
                        minIndex = i;
                    }
                }

                frequencies[minIndex]++;
                totalSymbols++;
            }
        }

        BuildCumulativeFrequencies();
    }

    return symbol;
}
```

`\compression\ZipLab\main.cpp`：

```cpp
#include "rANS.h"
#include <stdio.h>

// Simple file I/O functions without using STL
unsigned char* ReadFile(const char* filename, unsigned int* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s for reading\n", filename);
        return 0;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for file data
    unsigned char* data = new unsigned char[*size];

    // Read file data
    fread(data, 1, *size, file);

    fclose(file);
    return data;
}

bool WriteFile(const char* filename, unsigned char* data, unsigned int size) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Could not open file %s for writing\n", filename);
        return false;
    }

    // Write file data
    fwrite(data, 1, size, file);

    fclose(file);
    return true;
}

// Compress a file using rANS
bool CompressFile(const char* inputFilename, const char* outputFilename) {
    // Read input file
    unsigned int inputSize = 0;
    unsigned char* inputData = ReadFile(inputFilename, &inputSize);
    if (!inputData) {
        return false;
    }

    // Create output stream
    BitStream outputStream;

    // Write input file size to output
    outputStream.WriteInt32(inputSize);

    // Create encoder
    RansEncoder encoder(&outputStream);

    // Encode each byte
    for (unsigned int i = 0; i < inputSize; i++) {
        encoder.Encode(inputData[i]);
    }

    // Finalize encoding
    encoder.Finalize();

    // Write compressed data to output file
    bool success = WriteFile(outputFilename, outputStream.GetBuffer(), outputStream.GetSize());

    // Clean up
    delete[] inputData;

    return success;
}

// Decompress a file using rANS
bool DecompressFile(const char* inputFilename, const char* outputFilename) {
    // Read input file
    unsigned int inputSize = 0;
    unsigned char* inputData = ReadFile(inputFilename, &inputSize);
    if (!inputData) {
        return false;
    }

    // Create input stream
    BitStream inputStream(inputData, inputSize);

    // Read original file size
    unsigned int originalSize = inputStream.ReadInt32();

    // Create output buffer
    unsigned char* outputData = new unsigned char[originalSize];

    // Create decoder
    RansDecoder decoder(&inputStream);

    // Initialize decoder
    decoder.Initialize();

    // Decode each byte
    for (unsigned int i = 0; i < originalSize; i++) {
        outputData[i] = decoder.Decode();
    }

    // Write decompressed data to output file
    bool success = WriteFile(outputFilename, outputData, originalSize);

    // Clean up
    delete[] inputData;
    delete[] outputData;

    return success;
}

// Print usage information
void PrintUsage() {
    printf("ZipLab - rANS Compression Tool\n");
    printf("Usage:\n");
    printf("  ziplab c <input_file> <output_file> - Compress a file\n");
    printf("  ziplab d <input_file> <output_file> - Decompress a file\n");
}

// Main function
int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 4) {
        PrintUsage();
        return 1;
    }

    // Get operation type
    char operation = argv[1][0];

    // Get input and output filenames
    const char* inputFilename = argv[2];
    const char* outputFilename = argv[3];

    // Perform operation
    bool success = false;

    if (operation == 'c' || operation == 'C') {
        // Compress
        printf("Compressing %s to %s...\n", inputFilename, outputFilename);
        success = CompressFile(inputFilename, outputFilename);
    } else if (operation == 'd' || operation == 'D') {
        // Decompress
        printf("Decompressing %s to %s...\n", inputFilename, outputFilename);
        success = DecompressFile(inputFilename, outputFilename);
    } else {
        // Invalid operation
        PrintUsage();
        return 1;
    }

    // Check result
    if (success) {
        printf("Operation completed successfully.\n");
        return 0;
    } else {
        printf("Operation failed.\n");
        return 1;
    }
}
```

#### 实现说明

这个 rANS 压缩算法实现包含以下几个关键部分：

1. **BitStream 类**：

    自定义的二进制流类，提供了位级和字节级的读写操作，不依赖 STL。

2. **RansEncoder 类**：实现了 rANS 编码算法，包括：

   - 动态频率估算和更新
   - 自适应压缩
   - 64 位状态变量
   - 频率表正规化

3. **RansDecoder 类**：实现了 rANS 解码算法，与编码器对应。

4. **主程序**：提供了简单的文件压缩和解压缩功能。

##### 算法特点

- 使用 64 位整型实现 rANS 算法
- 动态频率估算，可以适应不同的数据特性
- 不使用 C++ STL，所有数据结构都是自定义的
- 包含详细的英文注释，解释了算法的关键步骤

##### 使用方法

编译后，可以通过命令行使用这个程序：

```bash
ZipLab.exe c input.txt compressed.bin  # 压缩文件
ZipLab.exe d compressed.bin output.txt # 解压文件
```

这个实现提供了一个完整的 rANS 压缩算法框架，您可以根据需要进一步优化或扩展它。
