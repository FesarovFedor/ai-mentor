//! Точка входа Tauri-приложения "ИИ-наставник".
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    mentor_tauri_lib::run()
}
