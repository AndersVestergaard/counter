import * as React from 'react'
import classes from './index.module.css'
import { useLocalStorage } from '../../components/hooks/use-local-storage'

export function Counter() {
  const [sumStr, setSumStr] = useLocalStorage({ key: 'sum', defaultValue: '0' })
  const invested = 200_000
  const increment = (invested * 0.07) / 365 / 24 / 60 / 60 / timesPerSecond

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

const timesPerSecond = 2
