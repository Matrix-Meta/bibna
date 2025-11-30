// src/tokenizer.hpp
#pragma once
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <set> // for checking if token is special

class SimpleTokenizer
{
  public:
    std::map<std::string, int> token_to_id;
    std::vector<std::string> id_to_token;
    int vocab_size = 0;
    
    // 特殊令牌 (例如，未知令牌的佔位符)
    // 我們可以將 0-255 保留給位元組令牌，因此特殊令牌從 256 開始。
    static const int UNKNOWN_TOKEN_ID = 256; 
    
    SimpleTokenizer()
    {
        // 預留 ID 0-255 給位元組回退
        // 這樣每個位元組都可以表示為一個令牌
        for (int i = 0; i < 256; ++i) {
            std::string byte_token = "<byte_" + std::to_string(i) + ">";
            token_to_id[byte_token] = i;
            id_to_token.push_back(byte_token);
        }
        vocab_size = 256; // 從 256 開始用於實際的詞彙

        // 加入 UNK 令牌
        if (token_to_id.find("<unk>") == token_to_id.end()) {
            token_to_id["<unk>"] = vocab_size;
            id_to_token.push_back("<unk>");
            vocab_size++;
        }
    }

    // 輔助：判斷UTF-8字元長度
    int get_utf8_char_length(unsigned char ch)
    {
        if ((ch & 0x80) == 0)
            return 1; // ASCII: 0xxxxxxx
        else if ((ch & 0xE0) == 0xC0)
            return 2; // 110xxxxx
        else if ((ch & 0xF0) == 0xE0)
            return 3; // 1110xxxx(中文常見)
        else if ((ch & 0xF8) == 0xF0)
            return 4; // 11110xxx
        return 1;     // 預設為1字元長度
    }

    // 將輸入字串切分為UTF-8字元
    std::vector<std::string> split_utf8(const std::string &text)
    {
        std::vector<std::string> chars;
        size_t i = 0;
        while (i < text.size())
        {
            int char_len = get_utf8_char_length(static_cast<unsigned char>(text[i]));
            if (i + char_len > text.size())
                break; // 防止越界
            chars.push_back(text.substr(i, char_len));
            i += char_len;
        }
        return chars;
    }

    // 建立詞彙表
    void build_vocab(const std::string &texts)
    {
        // 先將所有位元組令牌加入
        // 然後加入從文本中學習到的令牌
        // 確保從 256 開始分配新的 ID
        int current_id_counter = vocab_size; // Start after byte tokens and UNK
        
        auto chars = split_utf8(texts);
        for (const auto &ch : chars)
        {
            if (token_to_id.find(ch) == token_to_id.end())
            {
                token_to_id[ch] = current_id_counter;
                id_to_token.push_back(ch); // push_back will add to end
                current_id_counter++;
            }
        }
        vocab_size = current_id_counter; // Update final vocab size
        std::cout << "Tokenizer built size: " << vocab_size << "\n" << std::endl;
    }

    // 將字串編碼為ID序列
    std::vector<int> encode(const std::string &text)
    {
        std::vector<int> ids;
        auto chars = split_utf8(text);
        for (const auto &c : chars)
        {
            if (token_to_id.count(c))
            {
                ids.push_back(token_to_id[c]);
            }
            else
            {
                // 未知字元處理：位元組回退
                for (unsigned char byte_val : c) {
                    ids.push_back(byte_val); // 位元組值就是它的 ID (0-255)
                }
            }
        }
        return ids;
    }

    // 將ID序列解碼為字串
    std::string decode(const std::vector<int> &ids)
    {
        std::stringstream ss;
        for (int id : ids)
        {
            if (id >= 0 && id < 256) { // 如果是位元組令牌
                ss << static_cast<char>(id);
            }
            else if (id >= 256 && id < vocab_size) // 正常詞彙令牌
            {
                ss << id_to_token[id];
            } else {
                // 如果 ID 超出範圍，或者是一個我們不認識的特殊令牌 ID
                // 這裡我們直接忽略，或者可以替換為 "<unk>"
                ss << "?"; // 佔位符
            }
        }
        return ss.str();
    }

    // 儲存/載入Tokenizer
    void save_vocab(const std::string &path)
    {
        std::ofstream out(path);
        // 跳過位元組令牌 (0-255) 和 <unk>
        for (size_t i = UNKNOWN_TOKEN_ID + 1; i < id_to_token.size(); ++i) // Start after <unk>
            out << id_to_token[i] << "\n";
        out.close();
    }

    void load(const std::string &path)
    {
        std::ifstream in(path);
        std::string line;
        
        // 清除現有詞彙表並重新初始化位元組令牌和 <unk>
        token_to_id.clear();
        id_to_token.clear();
        vocab_size = 0;
        
        for (int i = 0; i < 256; ++i) {
            std::string byte_token = "<byte_" + std::to_string(i) + ">";
            token_to_id[byte_token] = i;
            id_to_token.push_back(byte_token);
        }
        vocab_size = 256; 
        
        if (token_to_id.find("<unk>") == token_to_id.end()) {
            token_to_id["<unk>"] = vocab_size;
            id_to_token.push_back("<unk>");
            vocab_size++;
        }

        while (std::getline(in, line))
        {
            if (!line.empty())
            {
                if (token_to_id.find(line) == token_to_id.end()) { // 避免重複添加特殊令牌
                    token_to_id[line] = vocab_size;
                    id_to_token.push_back(line);
                    vocab_size++;
                }
            }
        }
        in.close();
        std::cout << "Tokenizer loaded size: " << vocab_size << "\n" << std::endl;
    }
};