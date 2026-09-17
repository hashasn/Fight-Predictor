import { useState, useEffect } from 'react';


function App() {
  const [fights, setFights] = useState([]);
  useEffect(() => {
    fetch('http://127.0.0.1:8000/predictions/upcoming')
      .then(response => response.json())
      .then(data => setFights(data))
  }, [])
  return (
    <div>
      <h1>upcoming fights: {fights.length}</h1>
      {fights.map(fight => (
        <p key={fight.fighter_a}>
          {fight.fighter_a} vs {fight.fighter_b}
        </p>
      ))}
    </div>
  )
}

export default App;
