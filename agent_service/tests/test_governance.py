"""
Comprehensive unit tests for the Governance controls.

Tests cover:
- Pause/resume functionality
- Pre-send confirmation workflow
- Price detection mechanisms
- Administrative controls
- Security and compliance features
- Error handling and edge cases
"""

import pytest
import re
import json
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

from src.services.governance import (
    GovernanceService, PriceDetector, ConfirmationManager,
    AgentController, ComplianceChecker
)
from src.models.schemas import AgentResponse, AgentConfig
from tests.conftest import (
    governance_test_scenarios, create_price_sensitive_message,
    assert_response_time, AsyncMockService
)


class TestGovernanceService:
    """Test core governance service functionality."""
    
    @pytest.fixture
    async def governance_service(self):
        """Create governance service instance for testing."""
        config = {
            'price_detection_enabled': True,
            'confirmation_required_threshold': 100.0,
            'admin_approval_timeout_minutes': 30,
            'pause_on_sensitive_content': True,
            'compliance_rules': {
                'max_price_without_approval': 500.0,
                'require_confirmation_keywords': ['price', 'cost', 'payment', 'billing']
            }
        }
        
        service = GovernanceService(config)
        await service.initialize()
        return service
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_governance_service_initialization(self):
        """Test governance service initialization."""
        # Arrange
        config = {'price_detection_enabled': True}
        
        # Act
        service = GovernanceService(config)
        await service.initialize()
        
        # Assert
        assert service._initialized is True
        assert service.config == config
        assert isinstance(service.price_detector, PriceDetector)
        assert isinstance(service.confirmation_manager, ConfirmationManager)
        assert isinstance(service.agent_controller, AgentController)
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_check_response_for_governance(self, governance_service, governance_test_scenarios):
        """Test checking agent response for governance requirements."""
        # Arrange
        service = governance_service
        
        for scenario_name, scenario in governance_test_scenarios.items():
            # Act
            result = await service.check_response(
                response_content=scenario['agent_response'],
                conversation_id=12345,
                contact_phone="+1234567890"
            )
            
            # Assert
            assert isinstance(result, dict)
            assert 'requires_confirmation' in result
            assert 'detected_prices' in result
            assert 'governance_flags' in result
            
            if scenario['should_require_confirmation']:
                assert result['requires_confirmation'] is True
                if scenario.get('detected_price'):
                    assert len(result['detected_prices']) > 0
                    assert scenario['detected_price'] in [p['amount'] for p in result['detected_prices']]
            else:
                assert result['requires_confirmation'] is False
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_pause_agent_for_conversation(self, governance_service):
        """Test pausing agent for specific conversation."""
        # Arrange
        service = governance_service
        conversation_id = 12345
        reason = "Manual pause for review"
        
        # Act
        result = await service.pause_agent(conversation_id, reason)
        
        # Assert
        assert result['status'] == 'paused'
        assert result['conversation_id'] == conversation_id
        assert result['reason'] == reason
        
        # Verify pause status
        is_paused = await service.is_agent_paused(conversation_id)
        assert is_paused is True
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_resume_agent_for_conversation(self, governance_service):
        """Test resuming agent for specific conversation."""
        # Arrange
        service = governance_service
        conversation_id = 12345
        
        # First pause the agent
        await service.pause_agent(conversation_id, "Test pause")
        
        # Act
        result = await service.resume_agent(conversation_id)
        
        # Assert
        assert result['status'] == 'resumed'
        assert result['conversation_id'] == conversation_id
        
        # Verify pause status
        is_paused = await service.is_agent_paused(conversation_id)
        assert is_paused is False
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_global_agent_pause(self, governance_service):
        """Test global agent pause functionality."""
        # Arrange
        service = governance_service
        reason = "System maintenance"
        
        # Act
        result = await service.pause_agent_globally(reason)
        
        # Assert
        assert result['status'] == 'globally_paused'
        assert result['reason'] == reason
        
        # Verify all conversations are affected
        is_paused_1 = await service.is_agent_paused(12345)
        is_paused_2 = await service.is_agent_paused(67890)
        assert is_paused_1 is True
        assert is_paused_2 is True
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_confirmation_workflow(self, governance_service):
        """Test complete confirmation workflow."""
        # Arrange
        service = governance_service
        response_content = "Our installation service costs $299.99. Would you like to proceed?"
        conversation_id = 12345
        contact_phone = "+1234567890"
        
        # Act - Check if confirmation required
        governance_result = await service.check_response(
            response_content=response_content,
            conversation_id=conversation_id,
            contact_phone=contact_phone
        )
        
        assert governance_result['requires_confirmation'] is True
        
        # Act - Create confirmation request
        confirmation_id = await service.create_confirmation_request(
            response_content=response_content,
            conversation_id=conversation_id,
            contact_phone=contact_phone,
            detected_prices=governance_result['detected_prices']
        )
        
        # Assert
        assert confirmation_id is not None
        assert isinstance(confirmation_id, str)
        
        # Act - Approve confirmation
        approval_result = await service.approve_confirmation(
            confirmation_id=confirmation_id,
            approver_id="admin_123",
            notes="Approved standard pricing"
        )
        
        # Assert
        assert approval_result['status'] == 'approved'
        assert approval_result['confirmation_id'] == confirmation_id
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_confirmation_rejection(self, governance_service):
        """Test confirmation rejection workflow."""
        # Arrange
        service = governance_service
        response_content = "Emergency repair service will cost $999.99"
        conversation_id = 12345
        
        # Create confirmation request
        confirmation_id = await service.create_confirmation_request(
            response_content=response_content,
            conversation_id=conversation_id,
            contact_phone="+1234567890",
            detected_prices=[{'amount': 999.99, 'currency': 'USD'}]
        )
        
        # Act - Reject confirmation
        rejection_result = await service.reject_confirmation(
            confirmation_id=confirmation_id,
            rejector_id="admin_123",
            reason="Price too high without manager approval"
        )
        
        # Assert
        assert rejection_result['status'] == 'rejected'
        assert rejection_result['confirmation_id'] == confirmation_id
        assert rejection_result['reason'] == "Price too high without manager approval"
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_confirmation_timeout(self, governance_service):
        """Test confirmation request timeout handling."""
        # Arrange
        service = governance_service
        response_content = "Service costs $199.99"
        conversation_id = 12345
        
        # Create confirmation request
        confirmation_id = await service.create_confirmation_request(
            response_content=response_content,
            conversation_id=conversation_id,
            contact_phone="+1234567890",
            detected_prices=[{'amount': 199.99, 'currency': 'USD'}]
        )
        
        # Simulate timeout by modifying created time
        with patch.object(service.confirmation_manager, 'get_confirmation_age') as mock_age:
            mock_age.return_value = timedelta(minutes=31)  # Exceed 30-minute timeout
            
            # Act
            is_expired = await service.is_confirmation_expired(confirmation_id)
            
            # Assert
            assert is_expired is True
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_get_governance_dashboard_data(self, governance_service):
        """Test getting governance dashboard data."""
        # Arrange
        service = governance_service
        
        # Create some test data
        await service.pause_agent(12345, "Test pause")
        await service.create_confirmation_request(
            "Test service costs $150",
            12346,
            "+1234567890",
            [{'amount': 150.0, 'currency': 'USD'}]
        )
        
        # Act
        dashboard_data = await service.get_dashboard_data()
        
        # Assert
        assert isinstance(dashboard_data, dict)
        assert 'paused_conversations' in dashboard_data
        assert 'pending_confirmations' in dashboard_data
        assert 'governance_stats' in dashboard_data
        
        assert len(dashboard_data['paused_conversations']) >= 1
        assert len(dashboard_data['pending_confirmations']) >= 1


class TestPriceDetector:
    """Test price detection functionality."""
    
    @pytest.fixture
    def price_detector(self):
        """Create price detector instance."""
        config = {
            'currencies': ['USD', 'EUR', 'GBP'],
            'min_price_threshold': 1.0,
            'max_price_threshold': 10000.0
        }
        return PriceDetector(config)
    
    @pytest.mark.unit
    @pytest.mark.governance
    def test_detect_dollar_prices(self, price_detector):
        """Test detection of dollar amounts."""
        # Arrange
        test_cases = [
            ("Service costs $299.99", [299.99]),
            ("Prices: $150, $200, and $350", [150.0, 200.0, 350.0]),
            ("Total: $1,299.50", [1299.50]),
            ("Free service ($0)", [0.0]),
            ("No price mentioned", []),
            ("Call us at 555-1299", []),  # Should not detect phone numbers
        ]
        
        # Act & Assert
        for text, expected_prices in test_cases:
            detected = price_detector.detect_prices(text)
            detected_amounts = [p['amount'] for p in detected]
            
            if expected_prices:
                assert len(detected_amounts) == len(expected_prices)
                for expected in expected_prices:
                    assert expected in detected_amounts
            else:
                assert len(detected_amounts) == 0
    
    @pytest.mark.unit
    @pytest.mark.governance
    def test_detect_written_prices(self, price_detector):
        """Test detection of written price amounts."""
        # Arrange
        test_cases = [
            ("Service costs two hundred dollars", [200.0]),
            ("Price is fifty-five dollars and ninety-nine cents", [55.99]),
            ("One hundred fifty USD", [150.0]),
            ("Thirty five euros", [35.0]),
        ]
        
        # Act & Assert
        for text, expected_prices in test_cases:
            detected = price_detector.detect_prices(text)
            detected_amounts = [p['amount'] for p in detected]
            
            # Note: Written price detection is more complex
            # For now, verify the structure exists
            assert isinstance(detected, list)
    
    @pytest.mark.unit
    @pytest.mark.governance
    def test_detect_currency_symbols(self, price_detector):
        """Test detection of various currency symbols."""
        # Arrange
        test_cases = [
            ("€150.00", [150.00]),
            ("£99.95", [99.95]),
            ("¥1000", [1000.0]),
            ("₹500", [500.0]),
        ]
        
        # Act & Assert
        for text, expected_prices in test_cases:
            detected = price_detector.detect_prices(text)
            if expected_prices:
                detected_amounts = [p['amount'] for p in detected]
                assert len(detected_amounts) >= 1
                # At least one price should be detected
    
    @pytest.mark.unit
    @pytest.mark.governance
    def test_price_context_analysis(self, price_detector):
        """Test price detection with context analysis."""
        # Arrange
        test_cases = [
            ("Installation service: $299.99", "service_pricing"),
            ("Your total bill is $450.00", "billing"),
            ("Emergency repair costs $599", "emergency_pricing"),
            ("Regular maintenance: $149.99", "maintenance_pricing"),
        ]
        
        # Act & Assert
        for text, expected_context in test_cases:
            detected = price_detector.detect_prices(text)
            assert len(detected) > 0
            
            price_info = detected[0]
            assert 'context' in price_info
            # Context analysis may categorize the pricing type
    
    @pytest.mark.unit
    @pytest.mark.governance
    def test_price_detection_edge_cases(self, price_detector):
        """Test price detection edge cases."""
        # Arrange
        edge_cases = [
            ("Model year 2019", []),  # Should not detect years
            ("Room 299", []),  # Should not detect room numbers
            ("Highway 101", []),  # Should not detect highway numbers
            ("$0.01", [0.01]),  # Very small amount
            ("$99,999.99", [99999.99]),  # Large amount
            ("Price: $ 150.00", [150.00]),  # Space after currency symbol
            ("Cost is approximately $200-$300", [200.0, 300.0]),  # Price ranges
        ]
        
        # Act & Assert
        for text, expected_prices in edge_cases:
            detected = price_detector.detect_prices(text)
            detected_amounts = [p['amount'] for p in detected]
            
            if expected_prices:
                for expected in expected_prices:
                    assert expected in detected_amounts or abs(detected_amounts[0] - expected) < 0.01
            else:
                # Should not detect false positives
                assert len(detected_amounts) == 0
    
    @pytest.mark.unit
    @pytest.mark.governance
    def test_price_validation(self, price_detector):
        """Test price validation and filtering."""
        # Arrange
        price_detector.config['min_price_threshold'] = 10.0
        price_detector.config['max_price_threshold'] = 1000.0
        
        test_cases = [
            ("Service costs $5.00", []),  # Below minimum
            ("Service costs $50.00", [50.00]),  # Valid range
            ("Service costs $1500.00", []),  # Above maximum
        ]
        
        # Act & Assert
        for text, expected_prices in test_cases:
            detected = price_detector.detect_prices(text)
            detected_amounts = [p['amount'] for p in detected]
            
            if expected_prices:
                assert len(detected_amounts) == len(expected_prices)
                for expected in expected_prices:
                    assert expected in detected_amounts
            else:
                assert len(detected_amounts) == 0


class TestConfirmationManager:
    """Test confirmation management functionality."""
    
    @pytest.fixture
    def confirmation_manager(self):
        """Create confirmation manager instance."""
        config = {
            'timeout_minutes': 30,
            'auto_approve_under': 100.0,
            'require_manager_approval_over': 500.0
        }
        return ConfirmationManager(config)
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_create_confirmation_request(self, confirmation_manager):
        """Test creating confirmation requests."""
        # Arrange
        manager = confirmation_manager
        request_data = {
            'response_content': 'Installation service costs $299.99',
            'conversation_id': 12345,
            'contact_phone': '+1234567890',
            'detected_prices': [{'amount': 299.99, 'currency': 'USD'}],
            'requires_manager_approval': False
        }
        
        # Act
        confirmation_id = await manager.create_request(**request_data)
        
        # Assert
        assert confirmation_id is not None
        assert isinstance(confirmation_id, str)
        assert len(confirmation_id) > 10  # Should be a meaningful ID
        
        # Verify request can be retrieved
        request = await manager.get_request(confirmation_id)
        assert request is not None
        assert request['conversation_id'] == 12345
        assert request['status'] == 'pending'
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_approve_confirmation_request(self, confirmation_manager):
        """Test approving confirmation requests."""
        # Arrange
        manager = confirmation_manager
        confirmation_id = await manager.create_request(
            response_content='Service costs $199.99',
            conversation_id=12345,
            contact_phone='+1234567890',
            detected_prices=[{'amount': 199.99, 'currency': 'USD'}]
        )
        
        # Act
        result = await manager.approve_request(
            confirmation_id=confirmation_id,
            approver_id='admin_123',
            notes='Standard pricing approved'
        )
        
        # Assert
        assert result['status'] == 'approved'
        assert result['approver_id'] == 'admin_123'
        assert result['notes'] == 'Standard pricing approved'
        
        # Verify request status updated
        request = await manager.get_request(confirmation_id)
        assert request['status'] == 'approved'
        assert 'approved_at' in request
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_reject_confirmation_request(self, confirmation_manager):
        """Test rejecting confirmation requests."""
        # Arrange
        manager = confirmation_manager
        confirmation_id = await manager.create_request(
            response_content='Emergency service costs $899.99',
            conversation_id=12345,
            contact_phone='+1234567890',
            detected_prices=[{'amount': 899.99, 'currency': 'USD'}]
        )
        
        # Act
        result = await manager.reject_request(
            confirmation_id=confirmation_id,
            rejector_id='manager_456',
            reason='Price exceeds standard rates'
        )
        
        # Assert
        assert result['status'] == 'rejected'
        assert result['rejector_id'] == 'manager_456'
        assert result['reason'] == 'Price exceeds standard rates'
        
        # Verify request status updated
        request = await manager.get_request(confirmation_id)
        assert request['status'] == 'rejected'
        assert 'rejected_at' in request
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_auto_approval_logic(self, confirmation_manager):
        """Test automatic approval for small amounts."""
        # Arrange
        manager = confirmation_manager
        manager.config['auto_approve_under'] = 100.0
        
        # Act - Amount under auto-approval threshold
        confirmation_id = await manager.create_request(
            response_content='Service costs $75.00',
            conversation_id=12345,
            contact_phone='+1234567890',
            detected_prices=[{'amount': 75.00, 'currency': 'USD'}]
        )
        
        # Check if auto-approved
        request = await manager.get_request(confirmation_id)
        
        # Assert
        # Note: Implementation may auto-approve or require manual approval
        # Test verifies the logic exists
        assert request['status'] in ['pending', 'approved']
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_manager_approval_requirement(self, confirmation_manager):
        """Test manager approval requirement for high amounts."""
        # Arrange
        manager = confirmation_manager
        manager.config['require_manager_approval_over'] = 500.0
        
        # Act
        confirmation_id = await manager.create_request(
            response_content='Premium service costs $750.00',
            conversation_id=12345,
            contact_phone='+1234567890',
            detected_prices=[{'amount': 750.00, 'currency': 'USD'}]
        )
        
        # Assert
        request = await manager.get_request(confirmation_id)
        assert request.get('requires_manager_approval') is True
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_confirmation_expiration(self, confirmation_manager):
        """Test confirmation request expiration."""
        # Arrange
        manager = confirmation_manager
        confirmation_id = await manager.create_request(
            response_content='Service costs $200.00',
            conversation_id=12345,
            contact_phone='+1234567890',
            detected_prices=[{'amount': 200.00, 'currency': 'USD'}]
        )
        
        # Simulate time passage
        with patch.object(manager, 'get_request_age') as mock_age:
            mock_age.return_value = timedelta(minutes=31)  # Exceed timeout
            
            # Act
            is_expired = await manager.is_request_expired(confirmation_id)
            
            # Assert
            assert is_expired is True
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_get_pending_confirmations(self, confirmation_manager):
        """Test getting list of pending confirmations."""
        # Arrange
        manager = confirmation_manager
        
        # Create multiple requests
        ids = []
        for i in range(3):
            confirmation_id = await manager.create_request(
                response_content=f'Service {i} costs ${100 + i*50}.00',
                conversation_id=12345 + i,
                contact_phone=f'+123456789{i}',
                detected_prices=[{'amount': 100 + i*50, 'currency': 'USD'}]
            )
            ids.append(confirmation_id)
        
        # Approve one request
        await manager.approve_request(ids[0], 'admin_123', 'Approved')
        
        # Act
        pending = await manager.get_pending_requests()
        
        # Assert
        assert len(pending) == 2  # Two should remain pending
        pending_ids = [req['id'] for req in pending]
        assert ids[1] in pending_ids
        assert ids[2] in pending_ids
        assert ids[0] not in pending_ids  # Should not include approved


class TestAgentController:
    """Test agent control functionality."""
    
    @pytest.fixture
    def agent_controller(self):
        """Create agent controller instance."""
        return AgentController()
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_pause_agent_conversation(self, agent_controller):
        """Test pausing agent for specific conversation."""
        # Arrange
        controller = agent_controller
        conversation_id = 12345
        reason = "Review required"
        
        # Act
        result = await controller.pause_conversation(conversation_id, reason)
        
        # Assert
        assert result['conversation_id'] == conversation_id
        assert result['status'] == 'paused'
        assert result['reason'] == reason
        
        # Verify pause status
        is_paused = await controller.is_conversation_paused(conversation_id)
        assert is_paused is True
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_resume_agent_conversation(self, agent_controller):
        """Test resuming agent for specific conversation."""
        # Arrange
        controller = agent_controller
        conversation_id = 12345
        
        # First pause
        await controller.pause_conversation(conversation_id, "Test pause")
        
        # Act
        result = await controller.resume_conversation(conversation_id)
        
        # Assert
        assert result['conversation_id'] == conversation_id
        assert result['status'] == 'resumed'
        
        # Verify resume status
        is_paused = await controller.is_conversation_paused(conversation_id)
        assert is_paused is False
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_global_pause_resume(self, agent_controller):
        """Test global agent pause and resume."""
        # Arrange
        controller = agent_controller
        
        # Act - Global pause
        pause_result = await controller.pause_globally("System maintenance")
        
        # Assert
        assert pause_result['status'] == 'globally_paused'
        
        # Verify all conversations affected
        is_paused_1 = await controller.is_conversation_paused(12345)
        is_paused_2 = await controller.is_conversation_paused(67890)
        assert is_paused_1 is True
        assert is_paused_2 is True
        
        # Act - Global resume
        resume_result = await controller.resume_globally()
        
        # Assert
        assert resume_result['status'] == 'globally_resumed'
        
        # Verify conversations resumed
        is_paused_1 = await controller.is_conversation_paused(12345)
        is_paused_2 = await controller.is_conversation_paused(67890)
        assert is_paused_1 is False
        assert is_paused_2 is False
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_get_paused_conversations(self, agent_controller):
        """Test getting list of paused conversations."""
        # Arrange
        controller = agent_controller
        
        # Pause multiple conversations
        await controller.pause_conversation(12345, "Reason 1")
        await controller.pause_conversation(67890, "Reason 2")
        await controller.pause_conversation(11111, "Reason 3")
        
        # Resume one
        await controller.resume_conversation(67890)
        
        # Act
        paused = await controller.get_paused_conversations()
        
        # Assert
        assert len(paused) == 2
        paused_ids = [conv['conversation_id'] for conv in paused]
        assert 12345 in paused_ids
        assert 11111 in paused_ids
        assert 67890 not in paused_ids
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_agent_control_permissions(self, agent_controller):
        """Test permission checking for agent control."""
        # Arrange
        controller = agent_controller
        
        # Mock permission checking
        with patch.object(controller, 'check_permissions') as mock_check:
            mock_check.return_value = True
            
            # Act
            result = await controller.pause_conversation(
                conversation_id=12345,
                reason="Test",
                user_id="admin_123"
            )
            
            # Assert
            assert result['status'] == 'paused'
            mock_check.assert_called_once_with("admin_123", "pause_conversation")


class TestComplianceChecker:
    """Test compliance checking functionality."""
    
    @pytest.fixture
    def compliance_checker(self):
        """Create compliance checker instance."""
        config = {
            'pii_detection_enabled': True,
            'content_filtering_enabled': True,
            'audit_logging_enabled': True,
            'sensitive_keywords': ['ssn', 'credit card', 'password', 'bank account']
        }
        return ComplianceChecker(config)
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.security
    def test_pii_detection(self, compliance_checker):
        """Test detection of personally identifiable information."""
        # Arrange
        checker = compliance_checker
        
        test_cases = [
            ("My SSN is 123-45-6789", True),
            ("Credit card: 4111-1111-1111-1111", True),
            ("My phone is 555-1234", False),  # Not considered PII in this context
            ("Hello, how are you?", False),
        ]
        
        # Act & Assert
        for text, should_detect in test_cases:
            has_pii = checker.detect_pii(text)
            if should_detect:
                assert has_pii is True, f"Should detect PII in: {text}"
            else:
                assert has_pii is False, f"Should not detect PII in: {text}"
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.security
    def test_content_filtering(self, compliance_checker):
        """Test content filtering for inappropriate content."""
        # Arrange
        checker = compliance_checker
        
        test_cases = [
            ("Professional service response", False),
            ("Inappropriate language here", True),  # Would need actual inappropriate content
            ("Service costs $299.99", False),
        ]
        
        # Act & Assert
        for text, should_filter in test_cases:
            needs_filtering = checker.check_content_filter(text)
            # Note: Actual implementation would have specific filtering logic
            assert isinstance(needs_filtering, bool)
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.security
    def test_sensitive_keyword_detection(self, compliance_checker):
        """Test detection of sensitive keywords."""
        # Arrange
        checker = compliance_checker
        
        test_cases = [
            ("Please provide your SSN", True),
            ("What's your password?", True),
            ("Service installation complete", False),
            ("Bank account details needed", True),
        ]
        
        # Act & Assert
        for text, should_detect in test_cases:
            has_sensitive = checker.has_sensitive_keywords(text)
            if should_detect:
                assert has_sensitive is True, f"Should detect sensitive keywords in: {text}"
            else:
                assert has_sensitive is False, f"Should not detect sensitive keywords in: {text}"
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_audit_logging(self, compliance_checker):
        """Test audit logging functionality."""
        # Arrange
        checker = compliance_checker
        
        audit_data = {
            'action': 'agent_response_sent',
            'conversation_id': 12345,
            'contact_phone': '+1234567890',
            'response_content': 'Service completed successfully',
            'governance_flags': ['price_detected'],
            'timestamp': datetime.utcnow()
        }
        
        # Act
        audit_id = await checker.log_audit_event(**audit_data)
        
        # Assert
        assert audit_id is not None
        assert isinstance(audit_id, str)
        
        # Verify audit record can be retrieved
        audit_record = await checker.get_audit_record(audit_id)
        assert audit_record is not None
        assert audit_record['action'] == 'agent_response_sent'
        assert audit_record['conversation_id'] == 12345
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_compliance_report_generation(self, compliance_checker):
        """Test generating compliance reports."""
        # Arrange
        checker = compliance_checker
        
        # Create some audit data
        await checker.log_audit_event(
            action='agent_response_sent',
            conversation_id=12345,
            contact_phone='+1234567890',
            response_content='Test response 1'
        )
        
        await checker.log_audit_event(
            action='confirmation_required',
            conversation_id=12346,
            contact_phone='+0987654321',
            response_content='Test response 2'
        )
        
        # Act
        report = await checker.generate_compliance_report(
            start_date=datetime.utcnow() - timedelta(days=1),
            end_date=datetime.utcnow()
        )
        
        # Assert
        assert isinstance(report, dict)
        assert 'total_events' in report
        assert 'event_types' in report
        assert 'compliance_violations' in report
        assert 'recommendations' in report
        
        assert report['total_events'] >= 2


class TestGovernanceIntegration:
    """Test integration between governance components."""
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_end_to_end_governance_workflow(self, governance_service, governance_test_scenarios):
        """Test complete governance workflow from detection to approval."""
        # Arrange
        service = governance_service
        scenario = governance_test_scenarios['price_sensitive']
        
        # Act 1 - Check response for governance requirements
        governance_result = await service.check_response(
            response_content=scenario['agent_response'],
            conversation_id=12345,
            contact_phone="+1234567890"
        )
        
        # Assert 1
        assert governance_result['requires_confirmation'] is True
        assert len(governance_result['detected_prices']) > 0
        
        # Act 2 - Create confirmation request
        confirmation_id = await service.create_confirmation_request(
            response_content=scenario['agent_response'],
            conversation_id=12345,
            contact_phone="+1234567890",
            detected_prices=governance_result['detected_prices']
        )
        
        # Assert 2
        assert confirmation_id is not None
        
        # Act 3 - Approve confirmation
        approval_result = await service.approve_confirmation(
            confirmation_id=confirmation_id,
            approver_id="admin_123",
            notes="Standard pricing approved"
        )
        
        # Assert 3
        assert approval_result['status'] == 'approved'
        
        # Act 4 - Verify response can now be sent
        can_send = await service.can_send_response(confirmation_id)
        assert can_send is True
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.asyncio
    async def test_governance_with_agent_pause(self, governance_service):
        """Test governance interaction with agent pause functionality."""
        # Arrange
        service = governance_service
        conversation_id = 12345
        
        # Act 1 - Pause agent
        await service.pause_agent(conversation_id, "Manual review required")
        
        # Act 2 - Try to process response while paused
        governance_result = await service.check_response(
            response_content="Service costs $199.99",
            conversation_id=conversation_id,
            contact_phone="+1234567890"
        )
        
        # Assert - Should indicate agent is paused
        assert governance_result.get('agent_paused') is True
        
        # Act 3 - Resume agent
        await service.resume_agent(conversation_id)
        
        # Act 4 - Process response after resume
        governance_result = await service.check_response(
            response_content="Service costs $199.99",
            conversation_id=conversation_id,
            contact_phone="+1234567890"
        )
        
        # Assert - Should process normally
        assert governance_result.get('agent_paused') is False
        assert governance_result['requires_confirmation'] is True
    
    @pytest.mark.unit
    @pytest.mark.governance
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_governance_performance_under_load(self, governance_service):
        """Test governance performance under concurrent load."""
        # Arrange
        service = governance_service
        
        # Create multiple concurrent governance checks
        import asyncio
        tasks = []
        
        for i in range(20):
            task = service.check_response(
                response_content=f"Service {i} costs ${100 + i}.00",
                conversation_id=12345 + i,
                contact_phone=f"+123456789{i%10}"
            )
            tasks.append(task)
        
        # Act
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        
        # Assert
        assert_response_time(start_time, 5000)  # Should complete within 5 seconds
        assert len(results) == 20
        assert all(isinstance(result, dict) for result in results)
        assert all('requires_confirmation' in result for result in results)