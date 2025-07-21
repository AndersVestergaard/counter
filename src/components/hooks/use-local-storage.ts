/* eslint-env browser */
import * as React from 'react'

type LocalStorageProps = {
  readonly key: string
  readonly defaultValue: string | undefined | null
}

export function useLocalStorage({
  key,
  defaultValue,
}: LocalStorageProps): [string | undefined | null, SetValue] {
  const [localStorage, setLocalStorage] = React.useState<string | undefined | null>(() => {
    return getLocalStorageObject({ key, defaultValue })
  })

  React.useEffect(() => {
    if (localStorage) {
      window.localStorage.setItem(prefixKey(key), localStorage)
    }
  }, [localStorage, key])

  return [localStorage, setLocalStorage]
}

function getLocalStorageObject({
  key,
  defaultValue,
}: LocalStorageProps): string | undefined | null {
  const localStorageJson = window.localStorage.getItem(prefixKey(key))
  return localStorageJson ? localStorageJson : defaultValue
}

const prefixKey = (key: string): string => `counter-ui:${key}`

type SetValue = (
  value:
    | string
    | undefined
    | null
    | ((prev: string | undefined | null) => string | undefined | null)
) => void
