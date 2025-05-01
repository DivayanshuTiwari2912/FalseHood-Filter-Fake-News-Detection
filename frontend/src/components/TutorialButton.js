import React, { useState } from 'react';
import { FiHelpCircle } from 'react-icons/fi';
import TutorialModal from './TutorialModal';

// Help button component with tutorial modal
const TutorialButton = ({ pageName }) => {
  const [showTutorial, setShowTutorial] = useState(false);

  return (
    <>
      <button
        className="tutorial-help-btn"
        onClick={() => setShowTutorial(true)}
        title="Open interactive tutorial"
      >
        <FiHelpCircle />
      </button>
      
      <TutorialModal
        isOpen={showTutorial}
        onClose={() => setShowTutorial(false)}
        currentPage={pageName}
      />
    </>
  );
};

export default TutorialButton;