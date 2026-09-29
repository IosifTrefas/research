package org.example.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;

/***
 * @author: Iosif Trefas
 * @email: iosif.tre@gmail.com
 *
 * Health check and basic API endpoint controller
 */

@RestController
@RequestMapping("/health")
public class HealthController {

    @GetMapping
    public ResponseEntity<HealthResponse> health() {
        return ResponseEntity.ok(new HealthResponse("UP", "Application is running successfully"));
    }

    // Inner class for health response
    public static class HealthResponse {
        public String status;
        public String message;

        public HealthResponse(String status, String message) {
            this.status = status;
            this.message = message;
        }

        // Getters and setters
        public String getStatus() {
            return status;
        }

        public void setStatus(String status) {
            this.status = status;
        }

        public String getMessage() {
            return message;
        }

        public void setMessage(String message) {
            this.message = message;
        }
    }
}

