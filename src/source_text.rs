use crate::diagnostics::TextSpan;



#[derive(Clone, Debug)]
pub struct SourceText {
    pub lines: Vec<TextLine>,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct TextLine {
    pub text: SourceText,
    pub start: i32,
    pub len: i32,
    pub length_including: i32
}

impl SourceText {
    pub fn new(text: String) -> Self {
        let mut source_text = SourceText { lines: vec![], text: text.clone() };
        let mut lines = source_text.parse_lines(&source_text, text);
        source_text.lines = lines;
        source_text 
    }

    pub fn to_string(&self, start:usize, length:usize) -> String {
        let str: String = self.text.chars().skip(start).take(length).collect();
        str
    }

    pub fn get_line_index(&self, position:i32) -> usize {
        self.lines.clone().into_iter().position(|data| position >= data.start && position <= data.end()).unwrap_or_default()
    }

    pub fn parse_lines(&self, source_text: &SourceText, text: String) -> Vec<TextLine> {
        let mut result = vec![];

        let mut position = 0;
        let mut line_start = 0;

        while position < text.len() {
            let line_lenght = position - line_start;
            let line_break_widgeth = self.get_break_width(text.clone(), position as i32);
            if line_break_widgeth == 0 {
                position += 1;
            } else {
                let text_line = self.add_line(source_text, position as i32, line_start as i32, line_break_widgeth);
                result.push(text_line);
                position += line_break_widgeth as usize;
                line_start = position;
            }
        }

        if position > line_start {
            let text_line = self.add_line(source_text, position as i32, line_start as i32, 0);
            result.push(text_line);
        }

        return result;
    }

    pub fn add_line(&self, source_text: &SourceText, position:i32, line_start:i32, link_break_width:i32)  -> TextLine{
        let line_length = position - line_start;
        let line_length_including_line_break = line_length + link_break_width;
        let line = TextLine::new(source_text.clone(), line_start, line_length, line_length_including_line_break);
        line
    }

    pub fn get_break_width(&self, text: String, i:i32) -> i32 {
        let c = text.chars().nth(0);
        let l = if i + 1 >= text.len() as i32 {'\0'} else {text.chars().nth((i+1) as usize).unwrap_or_default()};

        if c == Some('\r') && l == '\n' {
            return 2
        }

        if c == Some('\r') || l == '\n' {
            return 1
        }

        0
    }
}


impl TextLine {
    pub fn new(source_text: SourceText, start: i32, len: i32, length_including: i32) -> Self {
        TextLine {
            text: source_text,
            start,
            len,
            length_including
        }
    }

    pub fn end(&self) -> i32 {
        self.start + self.len
    }

    pub fn span(&self) -> TextSpan {
        TextSpan::new(self.start, self.len)
    }
}