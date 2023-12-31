//
// Created by liangdaxin on 23-6-19.
//

#ifndef SGM_SGM_TYPES_H
#define SGM_SGM_TYPES_H
#include <iostream>
#include <limits>

constexpr auto Invalid_Float = std::numeric_limits<float>::infinity(); // 无效值

typedef int8_t			sint8;		// 有符号8位整数
typedef uint8_t			uint8;		// 无符号8位整数
typedef int16_t			sint16;		// 有符号16位整数
typedef uint16_t		uint16;		// 无符号16位整数
typedef int32_t			sint32;		// 有符号32位整数
typedef uint32_t		uint32;		// 无符号32位整数
typedef int64_t			sint64;		// 有符号64位整数
typedef uint64_t		uint64;		// 无符号64位整数
typedef float			float32;	// 单精度浮点
typedef double			float64;	// 双精度浮点

#endif //SGM_SGM_TYPES_H
