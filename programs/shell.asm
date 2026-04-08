; shell.asm — Minimal DOS shell for 8086
; Prints C:\> prompt, reads a line, loops forever.
; Pure 8086, no 186+ instructions.
;
; Build: nasm -f bin shell.asm -o shell.com

org 0x100

start:
    ; Print welcome banner
    mov dx, banner
    mov ah, 0x09
    int 0x21

prompt_loop:
    ; Print prompt "C:\> "
    mov dx, prompt
    mov ah, 0x09
    int 0x21

    ; Read a line of input into buffer
    mov dx, input_buf
    mov ah, 0x0A        ; DOS buffered input
    int 0x21

    ; Print newline
    mov dx, crlf
    mov ah, 0x09
    int 0x21

    ; Check if input is empty (just Enter)
    mov si, input_buf + 1
    cmp byte [si], 0
    je prompt_loop

    ; Check for "exit" command
    mov si, input_buf + 2
    cmp byte [si], 'e'
    jne .not_exit
    cmp byte [si+1], 'x'
    jne .not_exit
    cmp byte [si+2], 'i'
    jne .not_exit
    cmp byte [si+3], 't'
    jne .not_exit
    ; Exit
    int 0x20

.not_exit:
    ; Check for "ver" command
    cmp byte [si], 'v'
    jne .not_ver
    cmp byte [si+1], 'e'
    jne .not_ver
    cmp byte [si+2], 'r'
    jne .not_ver
    mov dx, ver_msg
    mov ah, 0x09
    int 0x21
    jmp prompt_loop

.not_ver:
    ; Check for "dir" command
    cmp byte [si], 'd'
    jne .not_dir
    cmp byte [si+1], 'i'
    jne .not_dir
    cmp byte [si+2], 'r'
    jne .not_dir
    mov dx, dir_msg
    mov ah, 0x09
    int 0x21
    jmp prompt_loop

.not_dir:
    ; Unknown command
    mov dx, bad_cmd
    mov ah, 0x09
    int 0x21
    jmp prompt_loop

; --- Data ---
banner:  db 'CSS-DOS', 0x0D, 0x0A
         db 'Type "ver", "dir", or "exit"', 0x0D, 0x0A, 0x0A, '$'
prompt:  db 'C:\> $'
crlf:    db 0x0D, 0x0A, '$'
ver_msg: db 'CSS-DOS version 1.0 (calcite)', 0x0D, 0x0A, '$'
dir_msg: db ' Volume in drive C has no label', 0x0D, 0x0A
         db ' Directory of C:\', 0x0D, 0x0A, 0x0A
         db 'KERNEL   SYS    45,868  01-25-2026', 0x0D, 0x0A
         db 'COMMAND  COM       ???  04-08-2026', 0x0D, 0x0A
         db '        2 file(s)', 0x0D, 0x0A, '$'
bad_cmd: db 'Bad command or file name', 0x0D, 0x0A, '$'

input_buf:
    db 80           ; max length
    db 0            ; actual length (filled by DOS)
    times 82 db 0   ; buffer
