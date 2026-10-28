/**
 * ==============================================================================
 * Cyber-Physical Deployment & Assistive Applications (Phase 4)
 * Production ESP32 Embedded BCI Actuator Controller
 * Target: ESP32 Dev Module / NodeMCU-32S
 * Baud Rate: 115200 bps
 * Protocol: 6-Byte Deterministic Binary UART Frame
 * Fail-Safe: Autonomous 500 ms Hardware/Software Watchdog Timer
 * ==============================================================================
 *
 * Frame Format:
 * [ Byte 0 ] 0xAA      : Frame Synchronization Preamble (High Byte)
 * [ Byte 1 ] 0x55      : Frame Synchronization Preamble (Low Byte)
 * [ Byte 2 ] CMD_ID    : Actuator Command (0x00=IDLE, 0x01=LEFT, 0x02=RIGHT, 0x03=FEET, 0x04=TONGUE/BRAKE)
 * [ Byte 3 ] CONF_BYTE : Quantized Confidence (0 - 255)
 * [ Byte 4 ] CRC8      : CRC-8 Checksum (Polynomial: 0x07, Init: 0x00) over [CMD_ID, CONF_BYTE]
 * [ Byte 5 ] 0x0D      : End-of-Frame Terminator ('\r')
 */

#include <Arduino.h>

// ==============================================================================
// Hardware Pin Definitions
// ==============================================================================

// Differential Motor Drive (H-Bridge L298N / TB6612FNG)
static const uint8_t PIN_MOTOR_LEFT_PWM  = 18;
static const uint8_t PIN_MOTOR_LEFT_DIR  = 19;
static const uint8_t PIN_MOTOR_RIGHT_PWM = 22;
static const uint8_t PIN_MOTOR_RIGHT_DIR = 23;

// Prosthetic Robotic Hand PWM
static const uint8_t PIN_SERVO_PWM       = 25;

// Status & Diagnostic Indicators
static const uint8_t PIN_STATUS_LED      = 2;   // On-board LED

// PWM Channels (ESP32 LEDC peripheral)
static const uint8_t PWM_CH_MOTOR_LEFT   = 0;
static const uint8_t PWM_CH_MOTOR_RIGHT  = 1;
static const uint8_t PWM_CH_SERVO        = 2;
static const uint32_t PWM_FREQ_MOTOR     = 20000; // 20 kHz ultrasonic motor PWM
static const uint32_t PWM_FREQ_SERVO     = 50;    // 50 Hz standard RC servo PWM
static const uint8_t PWM_RES_BITS        = 8;     // 8-bit resolution (0-255)

// ==============================================================================
// Protocol Constants and State Machine
// ==============================================================================

static const uint8_t PREAMBLE_BYTE_1   = 0xAA;
static const uint8_t PREAMBLE_BYTE_2   = 0x55;
static const uint8_t TERMINATOR_BYTE   = 0x0D;

enum BciCommand : uint8_t {
    CMD_IDLE         = 0x00,
    CMD_LEFT         = 0x01,
    CMD_RIGHT        = 0x02,
    CMD_FEET_FORWARD = 0x03,
    CMD_TONGUE_BRAKE = 0x04
};

enum ParserState : uint8_t {
    STATE_WAIT_HEADER_1 = 0,
    STATE_WAIT_HEADER_2,
    STATE_READ_CMD,
    STATE_READ_CONF,
    STATE_READ_CRC,
    STATE_WAIT_TAIL
};

// Fail-safe Watchdog Parameters
static const unsigned long WATCHDOG_TIMEOUT_MS = 500; // 500 ms autonomous timeout

// Parser runtime state variables
static ParserState parser_state = STATE_WAIT_HEADER_1;
static uint8_t rx_cmd = 0;
static uint8_t rx_conf = 0;
static uint8_t rx_crc = 0;

// System execution state variables
static unsigned long last_valid_packet_time = 0;
static bool watchdog_triggered = false;
static uint32_t valid_frames_count = 0;
static uint32_t corrupted_frames_count = 0;

// ==============================================================================
// CRC-8 Checksum Function (Polynomial: 0x07, Init: 0x00)
// ==============================================================================

uint8_t compute_crc8(const uint8_t *data, size_t len) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t j = 0; j < 8; j++) {
            if (crc & 0x80) {
                crc = ((crc << 1) ^ 0x07) & 0xFF;
            } else {
                crc = (crc << 1) & 0xFF;
            }
        }
    }
    return crc;
}

// ==============================================================================
// Actuator Actuation & Fail-Safe Primitives
// ==============================================================================

void apply_fail_safe_brake() {
    // Cut all motor outputs to neutral / brake immediately
    ledcWrite(PWM_CH_MOTOR_LEFT, 0);
    ledcWrite(PWM_CH_MOTOR_RIGHT, 0);
    digitalWrite(PIN_MOTOR_LEFT_DIR, LOW);
    digitalWrite(PIN_MOTOR_RIGHT_DIR, LOW);

    // Release prosthetic grip to neutral position (1.5 ms pulse = 19/255 at 50 Hz)
    ledcWrite(PWM_CH_SERVO, 19);

    // Fast blink indicator for watchdog condition
    digitalWrite(PIN_STATUS_LED, (millis() / 100) % 2);
}

void execute_actuation(uint8_t cmd, uint8_t confidence) {
    // Clear watchdog trip condition
    watchdog_triggered = false;
    digitalWrite(PIN_STATUS_LED, HIGH);

    uint8_t motor_power = confidence; // Direct proportional confidence scaling

    switch (cmd) {
        case CMD_IDLE:
            ledcWrite(PWM_CH_MOTOR_LEFT, 0);
            ledcWrite(PWM_CH_MOTOR_RIGHT, 0);
            digitalWrite(PIN_MOTOR_LEFT_DIR, LOW);
            digitalWrite(PIN_MOTOR_RIGHT_DIR, LOW);
            ledcWrite(PWM_CH_SERVO, 19); // Neutral grip
            break;

        case CMD_LEFT:
            // Differential turn left: Right motor forward, left motor idle
            digitalWrite(PIN_MOTOR_LEFT_DIR, LOW);
            digitalWrite(PIN_MOTOR_RIGHT_DIR, HIGH);
            ledcWrite(PWM_CH_MOTOR_LEFT, 0);
            ledcWrite(PWM_CH_MOTOR_RIGHT, motor_power);
            ledcWrite(PWM_CH_SERVO, 19);
            break;

        case CMD_RIGHT:
            // Differential turn right: Left motor forward, right motor idle
            digitalWrite(PIN_MOTOR_LEFT_DIR, HIGH);
            digitalWrite(PIN_MOTOR_RIGHT_DIR, LOW);
            ledcWrite(PWM_CH_MOTOR_LEFT, motor_power);
            ledcWrite(PWM_CH_MOTOR_RIGHT, 0);
            ledcWrite(PWM_CH_SERVO, 19);
            break;

        case CMD_FEET_FORWARD:
            // Drive straight forward with both motors
            digitalWrite(PIN_MOTOR_LEFT_DIR, HIGH);
            digitalWrite(PIN_MOTOR_RIGHT_DIR, HIGH);
            ledcWrite(PWM_CH_MOTOR_LEFT, motor_power);
            ledcWrite(PWM_CH_MOTOR_RIGHT, motor_power);
            ledcWrite(PWM_CH_SERVO, 19);
            break;

        case CMD_TONGUE_BRAKE:
            // Dynamic regenerative brake & prosthetic hand full grip closure
            digitalWrite(PIN_MOTOR_LEFT_DIR, LOW);
            digitalWrite(PIN_MOTOR_RIGHT_DIR, LOW);
            ledcWrite(PWM_CH_MOTOR_LEFT, 0);
            ledcWrite(PWM_CH_MOTOR_RIGHT, 0);
            // Grasp closed (2.0 ms pulse = 25/255 at 50 Hz)
            ledcWrite(PWM_CH_SERVO, 25);
            break;

        default:
            apply_fail_safe_brake();
            break;
    }
}

// ==============================================================================
// Arduino Setup & Main Loop
// ==============================================================================

void setup() {
    // Initialize high-speed UART for deterministic stream reception
    Serial.begin(115200);

    // Direction pins configuration
    pinMode(PIN_MOTOR_LEFT_DIR, OUTPUT);
    pinMode(PIN_MOTOR_RIGHT_DIR, OUTPUT);
    pinMode(PIN_STATUS_LED, OUTPUT);
    digitalWrite(PIN_MOTOR_LEFT_DIR, LOW);
    digitalWrite(PIN_MOTOR_RIGHT_DIR, LOW);
    digitalWrite(PIN_STATUS_LED, LOW);

    // Setup LEDC hardware PWM channels
    ledcSetup(PWM_CH_MOTOR_LEFT, PWM_FREQ_MOTOR, PWM_RES_BITS);
    ledcAttachPin(PIN_MOTOR_LEFT_PWM, PWM_CH_MOTOR_LEFT);

    ledcSetup(PWM_CH_MOTOR_RIGHT, PWM_FREQ_MOTOR, PWM_RES_BITS);
    ledcAttachPin(PIN_MOTOR_RIGHT_PWM, PWM_CH_MOTOR_RIGHT);

    ledcSetup(PWM_CH_SERVO, PWM_FREQ_SERVO, PWM_RES_BITS);
    ledcAttachPin(PIN_SERVO_PWM, PWM_CH_SERVO);

    apply_fail_safe_brake();
    last_valid_packet_time = millis();
}

void loop() {
    unsigned long current_time = millis();

    // 1. Non-blocking UART Stream Parser
    while (Serial.available() > 0) {
        uint8_t byte_in = Serial.read();

        switch (parser_state) {
            case STATE_WAIT_HEADER_1:
                if (byte_in == PREAMBLE_BYTE_1) {
                    parser_state = STATE_WAIT_HEADER_2;
                }
                break;

            case STATE_WAIT_HEADER_2:
                if (byte_in == PREAMBLE_BYTE_2) {
                    parser_state = STATE_READ_CMD;
                } else if (byte_in == PREAMBLE_BYTE_1) {
                    parser_state = STATE_WAIT_HEADER_2;
                } else {
                    parser_state = STATE_WAIT_HEADER_1;
                }
                break;

            case STATE_READ_CMD:
                rx_cmd = byte_in;
                parser_state = STATE_READ_CONF;
                break;

            case STATE_READ_CONF:
                rx_conf = byte_in;
                parser_state = STATE_READ_CRC;
                break;

            case STATE_READ_CRC:
                rx_crc = byte_in;
                parser_state = STATE_WAIT_TAIL;
                break;

            case STATE_WAIT_TAIL:
                if (byte_in == TERMINATOR_BYTE) {
                    // Compute CRC-8 over [rx_cmd, rx_conf]
                    uint8_t payload[2] = {rx_cmd, rx_conf};
                    uint8_t expected_crc = compute_crc8(payload, 2);

                    if (rx_crc == expected_crc) {
                        // Frame verified: execute actuation and refresh watchdog
                        last_valid_packet_time = current_time;
                        valid_frames_count++;
                        execute_actuation(rx_cmd, rx_conf);
                    } else {
                        corrupted_frames_count++;
                    }
                } else {
                    corrupted_frames_count++;
                }
                parser_state = STATE_WAIT_HEADER_1;
                break;
        }
    }

    // 2. Hardware Watchdog Fail-Safe Timer Enforcement
    if ((current_time - last_valid_packet_time) > WATCHDOG_TIMEOUT_MS) {
        if (!watchdog_triggered) {
            watchdog_triggered = true;
        }
        apply_fail_safe_brake();
    }
}
