import * as React from 'react'
import { DateTime } from 'luxon'
import classes from './index.module.css'
import { useLocalStorage } from '../../components/hooks/use-local-storage'

export function Counter() {
  const [sumStr, setSumStr] = useLocalStorage({ key: 'sum', defaultValue: '0' })
  const invested = 200_000
  const incrementPerSecond = (invested * 0.07) / 365 / 24 / 60 / 60
  const increment = incrementPerSecond / timesPerSecond

  React.useEffect(() => {
    const secSinceStart = calculateSecondsSinceNoonWhenStart()
    setSumStr((incrementPerSecond * secSinceStart).toFixed(10))
  }, [])

  React.useEffect(() => {
    const interval = setInterval(() => {
      setSumStr((prev) => (parseFloat(prev || '0') + increment).toFixed(10))
    }, 1000 / timesPerSecond)

    return () => clearInterval(interval)
  }, [increment])

  return (
    <div className={classes.container}>
      <div className={classes.numberDisplay}>{sumStr}</div>
    </div>
  )
}

function calculateSecondsSinceNoonWhenStart() {
  const start = DateTime.fromISO('2025-07-21T12:00:00Z')
  const now = DateTime.now()
  const diff = now.diff(start, 'seconds').seconds
  return Math.max(0, diff)
}

const timesPerSecond = 2
