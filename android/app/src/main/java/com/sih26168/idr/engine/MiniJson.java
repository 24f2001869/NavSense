package com.sih26168.idr.engine;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Lightweight zero-dependency JSON parser for verifying reference datasets in standalone JVM.
 */
public class MiniJson {

    public static Object parse(String jsonStr) {
        return new Parser(jsonStr).parseValue();
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> parseObject(String jsonStr) {
        return (Map<String, Object>) parse(jsonStr);
    }

    private static class Parser {
        private final String src;
        private int idx = 0;

        Parser(String src) {
            this.src = src;
        }

        private void skipWhitespace() {
            while (idx < src.length()) {
                char c = src.charAt(idx);
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    idx++;
                } else {
                    break;
                }
            }
        }

        Object parseValue() {
            skipWhitespace();
            if (idx >= src.length()) return null;
            char c = src.charAt(idx);
            if (c == '{') return parseObject();
            if (c == '[') return parseArray();
            if (c == '"') return parseString();
            if (c == 't' || c == 'f') return parseBoolean();
            if (c == 'n') return parseNull();
            return parseNumber();
        }

        private Map<String, Object> parseObject() {
            Map<String, Object> map = new HashMap<>();
            idx++; // skip '{'
            skipWhitespace();
            if (idx < src.length() && src.charAt(idx) == '}') {
                idx++;
                return map;
            }

            while (idx < src.length()) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                if (idx < src.length() && src.charAt(idx) == ':') {
                    idx++; // skip ':'
                }
                Object val = parseValue();
                map.put(key, val);
                skipWhitespace();
                if (idx < src.length() && src.charAt(idx) == ',') {
                    idx++;
                } else if (idx < src.length() && src.charAt(idx) == '}') {
                    idx++;
                    break;
                }
            }
            return map;
        }

        private List<Object> parseArray() {
            List<Object> list = new ArrayList<>();
            idx++; // skip '['
            skipWhitespace();
            if (idx < src.length() && src.charAt(idx) == ']') {
                idx++;
                return list;
            }

            while (idx < src.length()) {
                Object val = parseValue();
                list.add(val);
                skipWhitespace();
                if (idx < src.length() && src.charAt(idx) == ',') {
                    idx++;
                } else if (idx < src.length() && src.charAt(idx) == ']') {
                    idx++;
                    break;
                }
            }
            return list;
        }

        private String parseString() {
            idx++; // skip opening quote
            StringBuilder sb = new StringBuilder();
            while (idx < src.length()) {
                char c = src.charAt(idx++);
                if (c == '"') {
                    break;
                } else if (c == '\\') {
                    if (idx < src.length()) {
                        char esc = src.charAt(idx++);
                        if (esc == 'n') sb.append('\n');
                        else if (esc == 'r') sb.append('\r');
                        else if (esc == 't') sb.append('\t');
                        else sb.append(esc);
                    }
                } else {
                    sb.append(c);
                }
            }
            return sb.toString();
        }

        private Boolean parseBoolean() {
            if (src.startsWith("true", idx)) {
                idx += 4;
                return Boolean.TRUE;
            } else if (src.startsWith("false", idx)) {
                idx += 5;
                return Boolean.FALSE;
            }
            throw new IllegalArgumentException("Invalid boolean at position " + idx);
        }

        private Object parseNull() {
            if (src.startsWith("null", idx)) {
                idx += 4;
                return null;
            }
            throw new IllegalArgumentException("Invalid null at position " + idx);
        }

        private Number parseNumber() {
            int start = idx;
            if (idx < src.length() && (src.charAt(idx) == '-' || src.charAt(idx) == '+')) {
                idx++;
            }
            boolean isFloating = false;
            while (idx < src.length()) {
                char c = src.charAt(idx);
                if (Character.isDigit(c)) {
                    idx++;
                } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                    isFloating = true;
                    idx++;
                } else {
                    break;
                }
            }
            String numStr = src.substring(start, idx);
            if (isFloating) {
                return Double.parseDouble(numStr);
            } else {
                try {
                    return Long.parseLong(numStr);
                } catch (NumberFormatException e) {
                    return Double.parseDouble(numStr);
                }
            }
        }
    }
}
