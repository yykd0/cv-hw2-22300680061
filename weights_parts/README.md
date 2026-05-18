# HW2 model weights and tracking video parts

These files are split parts of HW2_model_best_weights_and_video_22300680061.zip.

Download all .partXX files, then merge them in order to recover the zip.

Windows PowerShell:

`powershell
Get-Content -Encoding Byte HW2_model_best_weights_and_video_22300680061.zip.part* | Set-Content -Encoding Byte HW2_model_best_weights_and_video_22300680061.zip
`

Linux/macOS:

`ash
cat HW2_model_best_weights_and_video_22300680061.zip.part* > HW2_model_best_weights_and_video_22300680061.zip
`

The merged zip contains best checkpoints for Task 1, Task 2 and Task 3, plus Task 2 tracking video and tracking summary files.
