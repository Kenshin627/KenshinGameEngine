#include "kenshin/util/BmpWriter.h"

#include <cstdio>
#include <cstring>

KENSHIN_NS_BEGIN

namespace util
{

    namespace
    {
#pragma pack(push, 1)
        struct BmpFileHeader
        {
            u16 type = 0x4D42;             // 'BM'
            u32 size = 0;
            u16 reserved0 = 0;
            u16 reserved1 = 0;
            u32 dataOffset = 0;
        };

        struct BmpInfoHeader
        {
            u32 size = sizeof(BmpInfoHeader);
            i32 width = 0;
            i32 height = 0;                // positive = bottom-up
            u16 planes = 1;
            u16 bitCount = 24;
            u32 compression = 0;           // BI_RGB
            u32 sizeImage = 0;
            i32 xPelsPerMeter = 2835;      // 72 DPI
            i32 yPelsPerMeter = 2835;
            u32 clrUsed = 0;
            u32 clrImportant = 0;
        };
#pragma pack(pop)
    } // namespace

    bool writeBmp24(const std::string& path, const std::vector<u8>& rgba,
                    u32 width, u32 height)
    {
        if (rgba.size() != static_cast<usize>(width) * height * 4)
            return false;

        const usize rowSize = (static_cast<usize>(width) * 3 + 3) & ~usize(3); // pad to 4
        const usize imageDataSize = rowSize * height;

        BmpFileHeader fileHeader;
        fileHeader.dataOffset = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader);
        fileHeader.size = static_cast<u32>(fileHeader.dataOffset + imageDataSize);

        BmpInfoHeader infoHeader;
        infoHeader.width = static_cast<i32>(width);
        infoHeader.height = static_cast<i32>(height);
        infoHeader.sizeImage = static_cast<u32>(imageDataSize);

        std::FILE* file = nullptr;
        if (fopen_s(&file, path.c_str(), "wb") != 0 || !file)
            return false;

        bool ok = std::fwrite(&fileHeader, sizeof(fileHeader), 1, file) == 1 &&
                  std::fwrite(&infoHeader, sizeof(infoHeader), 1, file) == 1;

        // BMP rows are bottom-up: write row y (top-down index) first for y = h-1 .. 0.
        std::vector<u8> row(rowSize, 0);
        for (u32 y = height; y-- > 0 && ok;)
        {
            const u8* src = rgba.data() + static_cast<usize>(y) * width * 4;
            for (u32 x = 0; x < width; ++x)
            {
                row[x * 3 + 0] = src[x * 4 + 2]; // B
                row[x * 3 + 1] = src[x * 4 + 1]; // G
                row[x * 3 + 2] = src[x * 4 + 0]; // R
            }
            ok = std::fwrite(row.data(), rowSize, 1, file) == 1;
        }

        std::fclose(file);
        return ok;
    }

} // namespace util

KENSHIN_NS_END
