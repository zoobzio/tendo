# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

1. **DO NOT** create a public GitHub issue
2. Email security details to the maintainers
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

## Security Best Practices

When using Tendo:

1. **Memory Management**: Tendo manages GPU and CPU memory through pools. Ensure proper cleanup with `Tensor.Free()` to avoid memory leaks.

2. **Numeric Precision**: Be aware that tensor operations use float32 by default. For applications requiring higher precision, consider validation of results.

3. **Backend Registration**: Custom backends should be registered carefully. Ensure backend implementations don't introduce security vulnerabilities.

4. **CUDA Operations**: When using CUDA backends, ensure proper device synchronization to avoid race conditions.

5. **Input Validation**: Validate tensor shapes and data before operations to prevent unexpected behavior.

## Security Features

Tendo is designed with security in mind:

- Thread-safe memory pooling with mutex protection
- Proper error handling with typed errors
- No network operations in core library
- No file system operations beyond normal Go imports
- Memory bounds checking in tensor operations

## Acknowledgments

We appreciate responsible disclosure of security vulnerabilities.
