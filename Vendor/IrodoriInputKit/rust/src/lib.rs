mod normalization;
use std::{
    ffi::{c_char, c_void, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr, slice, str,
};
use tokenizers::Tokenizer;

#[repr(C)]
#[derive(Default)]
pub struct IroTokens {
    pub data: *mut u32,
    pub len: usize,
    pub capacity: usize,
}

unsafe fn bytes<'a>(data: *const u8, len: usize) -> Result<&'a str, String> {
    if len == 0 {
        return Ok("");
    }
    if data.is_null() {
        return Err("null buffer".into());
    }
    str::from_utf8(slice::from_raw_parts(data, len)).map_err(|e| e.to_string())
}

unsafe fn error(target: *mut *mut c_char, message: String) {
    if !target.is_null() {
        let cleaned = message.replace('\0', " ");
        *target = CString::new(cleaned).expect("NUL removed").into_raw();
    }
}

#[no_mangle]
pub unsafe extern "C" fn iro_tokenizer_create(
    path: *const u8,
    len: usize,
    err: *mut *mut c_char,
) -> *mut c_void {
    if !err.is_null() {
        *err = ptr::null_mut();
    }
    match catch_unwind(AssertUnwindSafe(|| {
        let path = bytes(path, len)?;
        Tokenizer::from_bytes(path.as_bytes()).map_err(|e| e.to_string())
    })) {
        Ok(Ok(value)) => Box::into_raw(Box::new(value)).cast(),
        Ok(Err(message)) => {
            error(err, message);
            ptr::null_mut()
        }
        Err(_) => {
            error(err, "tokenizer creation panicked".into());
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn iro_tokenizer_encode(
    handle: *const c_void,
    text: *const u8,
    len: usize,
    out: *mut IroTokens,
    err: *mut *mut c_char,
) -> i32 {
    if !err.is_null() {
        *err = ptr::null_mut();
    }
    if out.is_null() {
        error(err, "null output".into());
        return 1;
    }
    *out = IroTokens::default();
    match catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() {
            return Err("null tokenizer".into());
        }
        let input = bytes(text, len)?;
        let tokenizer = &*handle.cast::<Tokenizer>();
        let encoding = tokenizer.encode(input, false).map_err(|e| e.to_string())?;
        Ok::<_, String>(encoding.get_ids().to_vec())
    })) {
        Ok(Ok(mut ids)) => {
            *out = IroTokens {
                data: ids.as_mut_ptr(),
                len: ids.len(),
                capacity: ids.capacity(),
            };
            std::mem::forget(ids);
            0
        }
        Ok(Err(message)) => {
            error(err, message);
            1
        }
        Err(_) => {
            error(err, "tokenizer encoding panicked".into());
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn iro_tokenizer_destroy(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle.cast::<Tokenizer>()));
    }
}
#[no_mangle]
pub unsafe extern "C" fn iro_tokens_free(tokens: IroTokens) {
    if !tokens.data.is_null() {
        drop(Vec::from_raw_parts(
            tokens.data,
            tokens.len,
            tokens.capacity,
        ));
    }
}
#[no_mangle]
pub unsafe extern "C" fn iro_error_free(message: *mut c_char) {
    if !message.is_null() {
        drop(CString::from_raw(message));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_utf8_is_rejected_without_losing_nul() {
        assert_eq!(
            unsafe { bytes(b"a\0b".as_ptr(), 3) }.unwrap().as_bytes(),
            b"a\0b"
        );
        assert!(unsafe { bytes([0xff].as_ptr(), 1) }.is_err());
        assert!(unsafe { bytes(ptr::null(), 1) }.is_err());
    }
    #[test]
    fn invalid_model_returns_owned_error() {
        let mut message = ptr::null_mut();
        let name = b"/not-an-irodori-model.json";
        let value = unsafe { iro_tokenizer_create(name.as_ptr(), name.len(), &mut message) };
        assert!(value.is_null());
        assert!(!message.is_null());
        unsafe {
            iro_error_free(message);
        }
    }
    #[test]
    fn null_handle_returns_empty_output_and_owned_error() {
        let mut message = ptr::null_mut();
        let mut result = IroTokens::default();
        assert_eq!(
            unsafe {
                iro_tokenizer_encode(ptr::null(), b"x".as_ptr(), 1, &mut result, &mut message)
            },
            1
        );
        assert!(result.data.is_null());
        assert!(!message.is_null());
        unsafe {
            iro_tokens_free(result);
            iro_error_free(message);
        }
    }
}
