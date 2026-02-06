use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

pub fn validate_xy_dimensions(shape: &[usize]) -> PyResult<()> {
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err("array must be 2D with shape (n, 2)"));
    }
    Ok(())
}

pub fn validate_point_dimensions(shape: &[usize]) -> PyResult<()> {
    if shape.len() != 1 || shape[0] != 2 {
        return Err(PyValueError::new_err("point array must have shape (2,)"));
    }
    Ok(())
}

pub fn validate_non_empty(shape: &[usize], name: &str) -> PyResult<()> {
    if shape.is_empty() || shape[0] == 0 {
        return Err(PyValueError::new_err(format!("{name} must be non-empty")));
    }
    Ok(())
}
