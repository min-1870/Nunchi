//! UI Automation, for one question: is the word we are about to replace right before the caret of the
//! focused field? Then Shift+Left selects exactly it, and the word can be typed over. Asked without sending
//! any key: in a terminal Shift+Left is not harmless (cmd starts its own selection and loses the next key).

use windows::core::Interface;
use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED};
use windows::Win32::UI::Accessibility::{
    CUIAutomation, IUIAutomation, IUIAutomation2, IUIAutomationElement, IUIAutomationTextPattern,
    IUIAutomationTextRange,
    TextPatternRangeEndpoint_End,
    TextPatternRangeEndpoint_Start, TextUnit_Character, UIA_DocumentControlTypeId, UIA_EditControlTypeId,
    UIA_TextPatternId,
};

pub struct Uia {
    auto: IUIAutomation,
    /// the element that had the focus when `focus_changed` last looked
    focused: Option<IUIAutomationElement>,
}

// The worker owns it and moves to its thread before making one: it is only ever made and used there.
unsafe impl Send for Uia {}

/// What the focused field shows before its caret.
#[derive(Debug, PartialEq)]
pub enum BeforeCaret {
    /// no text exposed, or no single empty selection (a caret) in it
    NoCaret,
    /// text that is not an edit field or document: a terminal's screen is static text (its caret is the
    /// shell's cursor, and Shift+Left there is the terminal's own selection)
    NotEditable,
    Text(String),
}

impl Uia {
    /// On the thread that will use it (COM, multithreaded).
    pub fn new() -> Option<Self> {
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok().ok()?;
            let auto: IUIAutomation = CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER).ok()?;
            // a hung app must not hold the correction for the default 2 s or more
            if let Ok(a2) = auto.cast::<IUIAutomation2>() {
                let _ = a2.SetConnectionTimeout(200);
                let _ = a2.SetTransactionTimeout(200);
            }
            Some(Uia { auto, focused: None })
        }
    }

    /// Another element has the focus than when this was last asked (or it cannot tell).
    pub fn focus_changed(&mut self) -> bool {
        unsafe {
            let Ok(now) = self.auto.GetFocusedElement() else {
                self.focused = None;
                return true;
            };
            let same = match &self.focused {
                Some(before) => self.auto.CompareElements(before, &now).is_ok_and(|b| b.as_bool()),
                None => false,
            };
            self.focused = Some(now);
            !same
        }
    }

    /// The focused element is a password field. None: cannot tell.
    pub fn is_password(&self) -> Option<bool> {
        unsafe { Some(self.auto.GetFocusedElement().ok()?.CurrentIsPassword().ok()?.as_bool()) }
    }

    /// Up to `n` characters before the caret of the focused field.
    pub fn before_caret(&self, n: usize) -> BeforeCaret {
        self.read(n).unwrap_or(BeforeCaret::NoCaret)
    }

    fn read(&self, n: usize) -> Option<BeforeCaret> {
        let r = match self.caret()? {
            Ok(r) => r,
            Err(()) => return Some(BeforeCaret::NotEditable),
        };
        unsafe {
            r.MoveEndpointByUnit(TextPatternRangeEndpoint_Start, TextUnit_Character, -(n as i32)).ok()?;
            Some(BeforeCaret::Text(r.GetText(-1).ok()?.to_string()))
        }
    }

    /// Up to `before` characters before the caret of the focused edit field and up to `after` after it. None:
    /// not an edit field, or no caret.
    pub fn around_caret(&self, before: usize, after: usize) -> Option<(String, String)> {
        let r = self.caret()?.ok()?;
        unsafe {
            let b = r.Clone().ok()?;
            b.MoveEndpointByUnit(TextPatternRangeEndpoint_Start, TextUnit_Character, -(before as i32)).ok()?;
            let before = b.GetText(-1).ok()?.to_string();
            // each side on its own: Chromium fails to read past the caret in some fields, and the words
            // before are worth having alone
            let after = r
                .MoveEndpointByUnit(TextPatternRangeEndpoint_End, TextUnit_Character, after as i32)
                .ok()
                .and_then(|_| r.GetText(-1).ok())
                .map(|t| t.to_string())
                .unwrap_or_default();
            Some((before, after))
        }
    }

    /// The text selected in the focused field (one selection). None: cannot tell.
    pub fn selection(&self) -> Option<String> {
        unsafe {
            let el = self.auto.GetFocusedElement().ok()?;
            let p: IUIAutomationTextPattern = el.GetCurrentPatternAs(UIA_TextPatternId).ok()?;
            let ranges = p.GetSelection().ok()?;
            if ranges.Length().ok()? != 1 {
                return None;
            }
            Some(ranges.GetElement(0).ok()?.GetText(-1).ok()?.to_string())
        }
    }

    /// The caret of the focused field, as an empty range. Some(Err): not an edit field or document.
    fn caret(&self) -> Option<Result<IUIAutomationTextRange, ()>> {
        unsafe {
            let el = self.auto.GetFocusedElement().ok()?;
            let kind = el.CurrentControlType().ok()?;
            if kind != UIA_EditControlTypeId && kind != UIA_DocumentControlTypeId {
                return Some(Err(()));
            }
            let p: IUIAutomationTextPattern = el.GetCurrentPatternAs(UIA_TextPatternId).ok()?;
            let ranges = p.GetSelection().ok()?;
            if ranges.Length().ok()? != 1 {
                return None;
            }
            let r = ranges.GetElement(0).ok()?;
            // a caret is an empty selection
            if r.CompareEndpoints(TextPatternRangeEndpoint_Start, &r, TextPatternRangeEndpoint_End).ok()? != 0 {
                return None;
            }
            Some(Ok(r))
        }
    }
}
