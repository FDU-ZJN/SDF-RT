#!/bin/bash
# 验证csrc目录结构

echo "Verifying csrc directory structure..."
echo "========================================"

# 检查目录
if [ ! -d "csrc/include" ]; then
    echo "❌ Missing: csrc/include"
    exit 1
fi

if [ ! -d "csrc/tests" ]; then
    echo "❌ Missing: csrc/tests"
    exit 1
fi

if [ ! -d "csrc/utils" ]; then
    echo "❌ Missing: csrc/utils"
    exit 1
fi

echo "✓ Directories created successfully"

# 检查头文件
files=("golden_model.h" "test_framework.h" "test_cases.h")
for file in "${files[@]}"; do
    if [ ! -f "csrc/include/$file" ]; then
        echo "❌ Missing: csrc/include/$file"
        exit 1
    fi
done
echo "✓ All header files created"

# 检查源文件
files=("golden_model.cpp" "test_framework.cpp" "test_cases.cpp" "main.cpp")
for file in "${files[@]}"; do
    if [ ! -f "csrc/$file" ]; then
        echo "❌ Missing: csrc/$file"
        exit 1
    fi
done
echo "✓ All source files created"

# 检查配置文件
if [ ! -f "csrc/Makefile" ]; then
    echo "❌ Missing: csrc/Makefile"
    exit 1
fi

if [ ! -f "csrc/README.md" ]; then
    echo "❌ Missing: csrc/README.md"
    exit 1
fi

if [ ! -f "csrc/ARCHITECTURE.md" ]; then
    echo "❌ Missing: csrc/ARCHITECTURE.md"
    exit 1
fi

echo "✓ All configuration files created"

# 检查代码行数
echo ""
echo "File statistics:"
echo "========================================"
for file in csrc/*.cpp csrc/*.h; do
    lines=$(wc -l < "$file")
    echo "$file: $lines lines"
done

echo ""
echo "========================================"
echo "✓ All checks passed!"
echo ""
echo "Next steps:"
echo "1. Make sure RayTriangleIntersection.sv exists in ../build/"
echo "2. Run: cd csrc && make"
echo "3. Run tests: make run"
