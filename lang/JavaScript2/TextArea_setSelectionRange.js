function setCss(opt) {
    var sr = document.getElementById("textarea-1");
    var len = sr.value.length;
    setSelectionRange(sr, len, len); // 将光标定位到文本最后
}

function setSelectionRange(input, selectionStart, selectionEnd) {
    if (input.setSelectionRange) {
        input.focus();
        input.setSelectionRange(selectionStart, selectionEnd);
    }
    else if (input.createTextRange) {
        var range = input.createTextRange();
        range.collapse(true);
        range.moveEnd('character', selectionEnd);
        range.moveStart('character', selectionStart);
        range.select();
    }
}