use crate::{bytes, error};
use std::{
    ffi::c_char,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
};
use unicode_normalization::UnicodeNormalization;

#[repr(C)]
#[derive(Default)]
pub struct IroBytes {
    pub data: *mut u8,
    pub len: usize,
    pub capacity: usize,
}

#[no_mangle]
pub unsafe extern "C" fn iro_nfkc(
    text: *const u8,
    len: usize,
    out: *mut IroBytes,
    err: *mut *mut c_char,
) -> i32 {
    if !err.is_null() {
        *err = ptr::null_mut();
    }
    if out.is_null() {
        error(err, "null output".into());
        return 1;
    }
    *out = IroBytes::default();
    match catch_unwind(AssertUnwindSafe(|| {
        Ok::<_, String>(bytes(text, len)?.nfkc().collect::<String>().into_bytes())
    })) {
        Ok(Ok(mut value)) => {
            *out = IroBytes {
                data: value.as_mut_ptr(),
                len: value.len(),
                capacity: value.capacity(),
            };
            std::mem::forget(value);
            0
        }
        Ok(Err(message)) => {
            error(err, message);
            1
        }
        Err(_) => {
            error(err, "normalization panicked".into());
            1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn iro_bytes_free(value: IroBytes) {
    if !value.data.is_null() {
        drop(Vec::from_raw_parts(value.data, value.len, value.capacity));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_nfkc_keeps_nul_and_unicode15_unassigned_scalars() {
        assert_eq!(unicode_normalization::UNICODE_VERSION, (15, 0, 0));
        let input = "Ａ\0\u{1ccd6}e\u{301}";
        let mut output = IroBytes::default();
        let mut message = ptr::null_mut();
        unsafe {
            assert_eq!(
                iro_nfkc(input.as_ptr(), input.len(), &mut output, &mut message),
                0
            );
            assert_eq!(
                std::slice::from_raw_parts(output.data, output.len),
                "A\0\u{1ccd6}é".as_bytes()
            );
            assert!(message.is_null());
            iro_bytes_free(output);
        }
    }
    #[test]
    fn rejects_invalid_utf8_and_null_output() {
        let mut output = IroBytes::default();
        let mut message = ptr::null_mut();
        unsafe {
            assert_eq!(iro_nfkc([0xff].as_ptr(), 1, &mut output, &mut message), 1);
            assert!(output.data.is_null());
            crate::iro_error_free(message);
            assert_eq!(iro_nfkc(ptr::null(), 0, ptr::null_mut(), &mut message), 1);
            crate::iro_error_free(message);
        }
    }
}
