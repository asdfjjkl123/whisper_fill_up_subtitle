# whisper_fill_up_subtitle
Using faster whisper to generate text base on ass subtitle time

使用faster whisper根據ass字幕行的時間進行音頻轉文字

## Feature 特點
 - Enter assb lines, whisper will generate the text for the lines.
 - 輸入ass空白行, whisper會生成對應行的文字
 - Audio is segmented based on the ASS line time to avoid long blank segments that allow the whisper to create text by itself.
 - 根據ass行時間切割音頻, 避免長空白段讓whisper自行創作
 - The number of sentences in the output file is the same as the number of ASS lines in the input file, so there will be no mismatch in sentence count due to Whisper's automatic sentence segmentation.
 - 輸出文件句子數與輸入文件的ass行數相同, 不會出現因whisper自動分句而句子數不對應

## Workflow Overview 流程
```
ASS Events (.txt)
        │
        ▼
Parse Dialogue timestamps 解析字幕行時間
        │
        ▼
FFmpeg split audio 根據時間切割音頻
        │
        ▼
1.wav 2.wav 3.wav ...
        │
        ▼
faster-whisper transcription 使用faster-whisper語音轉文字
        │
        ▼
1.txt 2.txt 3.txt ...
        │
        ▼
Merge results 合併whisper輸出
        │
        ├── TXT output
        └── ASS rebuilt subtitles
```

## Usage 用法
```bash
python whisper_fill_up_subtitle.py -i input.mkv -t events.txt
```
Output 輸出:
```
input_chunks/
input_subtitle.txt
```
Specify output path and whisper model dir and output in ass format

指定輸出路徑和whisper模型路徑, 並輸出為ass格式
```bash
python whisper_fill_up_subtitle.py -i input.mkv -t events.txt -o "D:\wav" --ass --whisper-output-dir "D:\txt" --model-dir "C:\Users\user\.cache\whisper"
```

An example batch file for dragging audio/video and events.txt (drag audio/video to bat), and output in ass format

範例bat檔用於拖放音頻/視頻到bat上開啟, 然後拖放events.txt, 並輸出為ass格式
```bash
@echo off
chcp 65001
:path
set PATH=%PATH%;C:\where\ffmpeg\in

set input="%~dpnx1"
set /p lines=ass line:

python whisper_fill_up_subtitle.py -i %input% -t %lines% -o "%~dp1wav" --ass --whisper-output-dir "%~dp1txt" --model-dir "C:\Users\user\.cache\whisper"

echo.
pause
```

For more usage 更多用法:
```bash
python whisper_fill_up_subtitle.py --help
```
### Example bat output 範例bat的輸出
```
C:\example\example.mp4 #input video 輸入用的視頻
C:\example\txt
 ├── 1.txt
 ├── 2.txt
 └── example_subtitle.txt
C:\example\wav
 ├── 1.wav
 └── 2.wav
```
## File example 文件格式範例
For events.txt #輸入events.txt
```
Dialogue: 0,0:00:19.00,0:00:20.90,Default,,0,0,0,,
Dialogue: 0,0:00:20.90,0:00:24.25,Default,,0,0,0,,
```
It is posible to use a complete ass file, but whisper ass lines including op, ed, screen text will take lots of time.

可以直接輸入完整ass檔案, 但使用whisper轉錄包括op, ed, 屏幕字行會花費大浪時間.

For output txt #whisper輸出txt範例

1.txt

```
あいうえお
```

2.txt
```
かきくけこ
さしすせそ
```
While merging txt file, txt with multiple lines will merge into single line, separate with a half-shaped space

當合併txt時, 若txt有多行, 會合併為一行, 使用半形空格分割

For example_subtitle.txt (no --ass) #最後輸出txt(非ass格式)
```
あいうえお
かきくけこ さしすせそ
```

example_subtitle.txt (with --ass) #最後輸出txt(ass格式)
```
Dialogue: 0,0:00:19.00,0:00:20.90,Default,,0,0,0,,あいうえお
Dialogue: 0,0:00:20.90,0:00:24.25,Default,,0,0,0,,かきくけこ さしすせそ
```

## Important 注意
While using default setting ```--model large-v3``` and ```--max-parallel 8```, the VRAM usage peak is around 10GB, please choose a suitable setting for yourself

當使用預設 ```--model large-v3```和 ```--max-parallel 8```, 顯存使用峰值約為10GB, 請自行選擇合適的參數
## Requirements 依賴項
 - Python ≥ 3.9

 - faster-whisper

 - whisper-ctranslate2

 - ffmpeg

Install dependencies 安裝依賴:

```bash
pip install faster-whisper
pip install whisper-ctranslate2
```
Install and add ffmpeg to PATH 下載並將ffmpeg放到PATH中:
https://ffmpeg.org/download.html

Verify:

```bash
ffmpeg -version
```
